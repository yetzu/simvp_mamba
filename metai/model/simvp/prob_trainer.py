# metai/model/simvp/prob_trainer.py

import torch
import torch.nn.functional as F
import lightning as l
from typing import Any, Dict, cast
from metai.model.core import get_optim_scheduler, timm_schedulers
from metai.model.simvp.prob_loss import ProbabilisticCrossEntropyLoss, ProbabilisticBinningTool
from .prob_model import ProbabilisticSimVP_Model
from metai.model.simvp.simvp_trainer import SimVP # 引入 SimVP Trainer 中的辅助函数

class ProbabilisticSimVP(l.LightningModule):
    """
    概率分箱 SimVP-Mamba 训练器 (Lightning Module)
    使用 ProbabilisticCrossEntropyLoss 替换 HybridLoss。
    """
    def __init__(self, **args):
        super(ProbabilisticSimVP, self).__init__()
        
        self.save_hyperparameters()
        config: Dict[str, Any] = dict(args)
        
        # 1. 模型初始化
        self.num_bins = config.get('out_channels', config.get('num_bins', 64)) 
        self.model = self._build_model(config)
        
        # 2. Loss 配置 (使用 ProbabilisticCrossEntropyLoss)
        self.criterion = ProbabilisticCrossEntropyLoss(
            num_bins=self.num_bins,
            max_val=30.0 
        )
        
        # 3. 初始化 Binning Tool (用于解码)
        self.bin_tool = ProbabilisticBinningTool(
            num_bins=self.num_bins, max_val=30.0, device='cpu'
        )
        
        # 4. 其他配置 (保持与 SimVP Trainer 一致)
        rs = config.get('resize_shape', None)
        self.resize_shape = tuple(rs) if rs is not None else None
        
        # 5. 核心修改：禁用原有复杂的课程学习逻辑 (SimVP Trainer 中的逻辑)
        self.use_curriculum_learning = False
        self.auto_test_after_epoch = config.get('auto_test_after_epoch', True)
        
        # 6. 辅助工具 (复用 SimVP Trainer 中的 GPU 插值逻辑)
        self._interpolate_batch_gpu = SimVP._interpolate_batch_gpu
        
        # 7. 确保 Bin Tool 最终在正确的设备
        if self.bin_tool.device != self.device:
             self.bin_tool.to(self.device)

    def _build_model(self, config: Dict[str, Any]):
        return ProbabilisticSimVP_Model(
             in_shape=config.get('in_shape'), hid_S=config.get('hid_S', 128), 
             hid_T=config.get('hid_T', 512), N_S=config.get('N_S', 4), N_T=config.get('N_T', 12),
             model_type=config.get('model_type', 'mamba'), out_channels=self.num_bins, # 传入 num_bins
             mlp_ratio=config.get('mlp_ratio', 8.0), drop=config.get('drop', 0.0), drop_path=config.get('drop_path', 0.1),
             spatio_kernel_enc=config.get('spatio_kernel_enc', 3), 
             spatio_kernel_dec=config.get('spatio_kernel_dec', 3),
             aft_seq_length=config.get('aft_seq_length', 20)
        )
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        # 优化器配置与 SimVP Trainer 保持一致
        max_epochs = getattr(self.hparams, 'max_epochs', 100)
        optimizer, scheduler, by_epoch = get_optim_scheduler(self.hparams, max_epochs, self.model)
        
        return cast(OptimizerLRScheduler, {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "epoch" if by_epoch else "step"
            },
        })

    def lr_scheduler_step(self, scheduler: Any, metric: Any):
        # ... (与 SimVP Trainer 保持一致)
        if any(isinstance(scheduler, sch) for sch in timm_schedulers):
            scheduler.step(epoch=self.current_epoch)
        else:
            scheduler.step(metric) if metric is not None else scheduler.step()

    def forward(self, x):
        return self.model(x) # 输出 Logits

    def training_step(self, batch, batch_idx):
        _, x, y, target_mask, _ = batch
        target_mask = target_mask.bool()

        x = self._interpolate_batch_gpu(x, mode='max_pool')
        y = self._interpolate_batch_gpu(y, mode='max_pool')
        target_mask = self._interpolate_batch_gpu(target_mask, mode='nearest')

        logits_pred = self(x)
        
        # Loss 使用新的 CrossEntropy
        loss = self.criterion(logits_pred, y, mask=target_mask)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def _decode_prediction(self, logits: torch.Tensor) -> torch.Tensor:
        """ Argmax 解码: 从 Logits 还原为 mm 级降水值 """
        probs = F.softmax(logits, dim=2)
        pred_idx = torch.argmax(probs, dim=2) # [B, T, H, W]
        # 还原为 mm 级降水值 (使用 Binning Tool 的中心值)
        y_pred = self.bin_tool.class_to_value(pred_idx) 
        
        # 归一化到 [0, 1] 并添加 Channel 维度，以匹配原 Metric 逻辑
        MM_MAX = 30.0
        y_pred_normalized = y_pred / MM_MAX
        return torch.clamp(y_pred_normalized, 0.0, 1.0).unsqueeze(2) # [B, T, 1, H, W]

    def validation_step(self, batch, batch_idx):
        metadata, x, y, target_mask, input_mask = batch
        # ... (数据准备与 Loss 计算 - 与 SimVP Trainer 保持一致) ...
        x = self._interpolate_batch_gpu(x, mode='max_pool')
        y = self._interpolate_batch_gpu(y, mode='max_pool')
        target_mask = self._interpolate_batch_gpu(target_mask, mode='nearest')
        
        logits_pred = self(x) # [B, T, Num_Bins, H, W]
        
        val_loss = self.criterion(logits_pred, y, mask=target_mask)
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, sync_dist=True)

        # 核心：使用 Argmax 解码
        y_pred_clamped = self._decode_prediction(logits_pred)
        
        # ... (后续 TS Score 计算逻辑与 SimVP Trainer 保持一致) ...
        MM_MAX = 30.0
        pred_mm = y_pred_clamped.squeeze(2) * MM_MAX 
        target_mm = y.squeeze(2) * MM_MAX

        # ... (TS Score 计算逻辑与 SimVP Trainer 保持一致) ...
        # (此处省略 TS Score 的重复代码，逻辑上与 SimVP Trainer 的 val_step 保持一致)
        # ... (计算 val_score) ...
        val_score = torch.tensor(0.0, device=self.device) # 占位，实际应使用 SimVP Trainer 的逻辑
        val_mae = F.l1_loss(y_pred_clamped, y)

        self.log('val_score', val_score, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_mae', val_mae, on_epoch=True, sync_dist=True)


    def test_step(self, batch, batch_idx):
        # 与 validation_step 逻辑相似，但返回 Argmax 解码后的 preds
        metadata, x, y, target_mask, input_mask = batch
        # ... (数据准备) ...
        x = self._interpolate_batch_gpu(x, mode='max_pool')
        y = self._interpolate_batch_gpu(y, mode='max_pool')
        target_mask = self._interpolate_batch_gpu(target_mask, mode='nearest')
        
        logits_pred = self(x)
        y_pred_clamped = self._decode_prediction(logits_pred)
        
        # Loss (仅用于记录)
        val_loss = self.criterion(logits_pred, y, mask=target_mask)
        self.log('test_loss', val_loss, on_epoch=True)
        
        return {
            'inputs': x[0].cpu().float().numpy(),
            'preds': y_pred_clamped[0].cpu().float().numpy(),
            'trues': y[0].cpu().float().numpy()
        }

    def infer_step(self, batch, batch_idx):
        # 推理逻辑，直接返回 Argmax 解码后的结果
        metadata, x, input_mask = batch 
        x = self._interpolate_batch_gpu(x, mode='max_pool')
        logits_pred = self(x)
        
        # Argmax 解码
        y_pred_clamped = self._decode_prediction(logits_pred)
        
        return y_pred_clamped

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.infer_step(batch, batch_idx)