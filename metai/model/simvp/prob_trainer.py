# metai/model/simvp/prob_trainer.py

import torch
import torch.nn.functional as F
import lightning as l
from typing import Any, Dict, cast
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from metai.model.core import get_optim_scheduler, timm_schedulers
from metai.model.simvp.prob_loss import ProbabilisticCrossEntropyLoss, ProbabilisticBinningTool
from .prob_model import ProbabilisticSimVP_Model

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
        
        # 4. 其他配置
        rs = config.get('resize_shape', None)
        self.resize_shape = tuple(rs) if rs is not None else None
        
        # 5. 核心修改：禁用原有复杂的课程学习逻辑
        self.use_curriculum_learning = False
        self.auto_test_after_epoch = config.get('auto_test_after_epoch', True)
        
        # [Fix] 移除错误的函数赋值，直接在该类中实现 _interpolate_batch_gpu
        # self._interpolate_batch_gpu = SimVP._interpolate_batch_gpu 
        
        # 7. 确保 Bin Tool 最终在正确的设备 (使用之前修复的 to 方法)
        if self.bin_tool.device != self.device:
             # 检查 bin_tool 是否有 to 方法 (防御性编程)
             if hasattr(self.bin_tool, 'to'):
                 self.bin_tool.to(self.device)

    def _build_model(self, config: Dict[str, Any]):
        return ProbabilisticSimVP_Model(
             in_shape=config.get('in_shape'), hid_S=config.get('hid_S', 128), 
             hid_T=config.get('hid_T', 512), N_S=config.get('N_S', 4), N_T=config.get('N_T', 12),
             model_type=config.get('model_type', 'mamba'), out_channels=self.num_bins,
             mlp_ratio=config.get('mlp_ratio', 8.0), drop=config.get('drop', 0.0), drop_path=config.get('drop_path', 0.1),
             spatio_kernel_enc=config.get('spatio_kernel_enc', 3), 
             spatio_kernel_dec=config.get('spatio_kernel_dec', 3),
             aft_seq_length=config.get('aft_seq_length', 20)
        )
    
    # [Fix] 直接在类中实现插值方法，避免 unbound method 错误
    def _interpolate_batch_gpu(self, batch_tensor: torch.Tensor, mode: str = 'max_pool') -> torch.Tensor:
        if self.resize_shape is None: return batch_tensor
        T, C, H, W = batch_tensor.shape[1:]
        target_H, target_W = self.resize_shape
        if H == target_H and W == target_W: return batch_tensor
        
        is_bool = batch_tensor.dtype == torch.bool
        if is_bool: batch_tensor = batch_tensor.float()
        
        B = batch_tensor.shape[0]
        batch_tensor = batch_tensor.view(B * T, C, H, W)
        
        if mode == 'max_pool':
            processed_tensor = F.adaptive_max_pool2d(batch_tensor, output_size=self.resize_shape) if target_H < H or target_W < W else F.interpolate(batch_tensor, size=self.resize_shape, mode='bilinear', align_corners=False)
        elif mode in ['nearest', 'bilinear']:
            align = False if mode == 'bilinear' else None
            processed_tensor = F.interpolate(batch_tensor, size=self.resize_shape, mode=mode, align_corners=align)
        else:
            raise ValueError(f"Unsupported interpolation mode: {mode}")

        processed_tensor = processed_tensor.view(B, T, C, target_H, target_W)
        if is_bool: processed_tensor = processed_tensor.bool()
        return processed_tensor

    def configure_optimizers(self) -> OptimizerLRScheduler:
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
        if any(isinstance(scheduler, sch) for sch in timm_schedulers):
            scheduler.step(epoch=self.current_epoch)
        else:
            scheduler.step(metric) if metric is not None else scheduler.step()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        _, x, y, target_mask, _ = batch
        target_mask = target_mask.bool()

        x = self._interpolate_batch_gpu(x, mode='max_pool')
        y = self._interpolate_batch_gpu(y, mode='max_pool')
        target_mask = self._interpolate_batch_gpu(target_mask, mode='nearest')
        
        MM_MAX_PHYSICAL = 30.0 
        y_for_binning = y * MM_MAX_PHYSICAL 

        logits_pred = self(x)
        loss = self.criterion(logits_pred, y_for_binning, mask=target_mask)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def _decode_prediction(self, logits: torch.Tensor) -> torch.Tensor:
        """ Argmax 解码 """
        probs = F.softmax(logits, dim=2)
        pred_idx = torch.argmax(probs, dim=2)
        
        # 确保 bin_tool 在正确设备
        if hasattr(self.bin_tool, 'to') and self.bin_tool.device != logits.device:
            self.bin_tool.to(logits.device)
            
        y_pred = self.bin_tool.class_to_value(pred_idx) 
        y_pred[y_pred < 0.05] = 0.0
        
        MM_MAX = 30.0
        y_pred_normalized = y_pred / MM_MAX
        return torch.clamp(y_pred_normalized, 0.0, 1.0).unsqueeze(2)

    def validation_step(self, batch, batch_idx):
        metadata, x, y, target_mask, input_mask = batch
        x = self._interpolate_batch_gpu(x, mode='max_pool')
        y = self._interpolate_batch_gpu(y, mode='max_pool')
        target_mask = self._interpolate_batch_gpu(target_mask, mode='nearest')
        
        logits_pred = self(x)
        
        MM_MAX_PHYSICAL = 30.0 
        y_for_binning = y * MM_MAX_PHYSICAL

        val_loss = self.criterion(logits_pred, y_for_binning, mask=target_mask)
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, sync_dist=True)

        y_pred_clamped = self._decode_prediction(logits_pred)
        
        val_mae = F.l1_loss(y_pred_clamped, y)
        val_score = torch.tensor(0.0, device=self.device) # 占位

        self.log('val_score', val_score, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_mae', val_mae, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        metadata, x, y, target_mask, input_mask = batch
        x = self._interpolate_batch_gpu(x, mode='max_pool')
        y = self._interpolate_batch_gpu(y, mode='max_pool')
        target_mask = self._interpolate_batch_gpu(target_mask, mode='nearest')
        
        logits_pred = self(x)
        y_pred_clamped = self._decode_prediction(logits_pred)
        
        MM_MAX_PHYSICAL = 30.0 
        y_for_binning = y * MM_MAX_PHYSICAL
        val_loss = self.criterion(logits_pred, y_for_binning, mask=target_mask)
        self.log('test_loss', val_loss, on_epoch=True)
        
        return {
            'inputs': x[0].cpu().float().numpy(),
            'preds': y_pred_clamped[0].cpu().float().numpy(),
            'trues': y[0].cpu().float().numpy()
        }

    def infer_step(self, batch, batch_idx):
        metadata, x, input_mask = batch 
        x = self._interpolate_batch_gpu(x, mode='max_pool')
        logits_pred = self(x)
        y_pred_clamped = self._decode_prediction(logits_pred)
        return y_pred_clamped

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.infer_step(batch, batch_idx)