# metai/model/simvp/prob_trainer.py

import torch
import torch.nn.functional as F
import lightning as l
from typing import Any, Dict, cast, Optional
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from metai.model.core import get_optim_scheduler, timm_schedulers
from metai.model.simvp.prob_loss import ProbabilisticCrossEntropyLoss, ProbabilisticBinningTool
from .prob_model import ProbabilisticSimVP_Model

class ProbabilisticSimVP(l.LightningModule):
    """
    概率分箱 SimVP-Mamba 训练器 (Lightning Module)
    
    该训练器用于训练基于概率分箱（Probabilistic Binning）的 SimVP 模型。
    它将回归问题转化为分类问题，通过预测降水值所属的区间（Bin）的概率分布来进行预测。
    
    主要特性：
    1. 集成 Gaussian Soft Labels 和 Focal Loss 的改进版 Loss。
    2. 支持 Temperature Scaling 的解码策略，用于控制预测分布的锐度。
    3. 支持迁移学习（Transfer Learning）和微调。
    """
    
    def __init__(self, **args):
        """
        初始化 ProbabilisticSimVP 训练器。
        
        Args:
            **args: 配置参数字典，包含模型结构、训练超参数和 Loss 参数。
                - num_bins (int): 概率分箱的数量，默认 40。
                - sigma (float): 高斯软标签的标准差，默认 2.0。
                - use_focal (bool): 是否启用 Focal Loss，默认 True。
                - gamma (float): Focal Loss 的聚焦参数，默认 2.0。
                - ... 其他 SimVP 模型参数 ...
        """
        super(ProbabilisticSimVP, self).__init__()
        
        self.save_hyperparameters()
        config: Dict[str, Any] = dict(args)
        
        # --- 1. 参数提取 ---
        self.num_bins = config.get('num_bins', 40)
        self.sigma = config.get('sigma', 2.0)
        self.use_focal = config.get('use_focal', True)
        self.gamma = config.get('gamma', 2.0)
        
        # 强制将 out_channels 设置为 num_bins，确保模型输出维度正确
        config['out_channels'] = self.num_bins
        
        # --- 2. 模型构建 ---
        self.model = self._build_model(config)
        
        # --- 3. Loss 函数配置 ---
        # 使用改进版 Loss：Gaussian Soft Labels + Focal Loss
        self.criterion = ProbabilisticCrossEntropyLoss(
            num_bins=self.num_bins,
            max_val=30.0,
            sigma=self.sigma,
            use_focal=self.use_focal,
            gamma=self.gamma
        )
        
        # --- 4. 工具类初始化 ---
        # 用于将类别索引还原为物理数值，以及辅助解码
        self.bin_tool = ProbabilisticBinningTool(
            num_bins=self.num_bins, 
            max_val=30.0, 
            device='cpu' # 初始在 CPU，使用时会自动同步设备
        )
        
        # --- 5. 其他配置 ---
        rs = config.get('resize_shape', None)
        self.resize_shape = tuple(rs) if rs is not None else None
        
        self.auto_test_after_epoch = config.get('auto_test_after_epoch', True)
        
        # 确保 Bin Tool 最终在正确的设备 (虽然 forward 中也有检查)
        if hasattr(self.bin_tool, 'to') and self.bin_tool.device != self.device:
             self.bin_tool.to(self.device)

    def _build_model(self, config: Dict[str, Any]) -> ProbabilisticSimVP_Model:
        """根据配置构建 ProbabilisticSimVP_Model 模型实例"""
        return ProbabilisticSimVP_Model(
             in_shape=config.get('in_shape'), 
             hid_S=config.get('hid_S', 128), 
             hid_T=config.get('hid_T', 512), 
             N_S=config.get('N_S', 4), 
             N_T=config.get('N_T', 12),
             model_type=config.get('model_type', 'mamba'), 
             out_channels=self.num_bins, # 确保传入 num_bins 作为输出通道数
             mlp_ratio=config.get('mlp_ratio', 8.0), 
             drop=config.get('drop', 0.0), 
             drop_path=config.get('drop_path', 0.1),
             spatio_kernel_enc=config.get('spatio_kernel_enc', 3), 
             spatio_kernel_dec=config.get('spatio_kernel_dec', 3),
             aft_seq_length=config.get('aft_seq_length', 20)
        )
    
    def _interpolate_batch_gpu(self, batch_tensor: torch.Tensor, mode: str = 'max_pool') -> torch.Tensor:
        """
        在 GPU 上对 Batch 数据进行插值/缩放。
        
        Args:
            batch_tensor: 输入张量 [B, T, C, H, W]
            mode: 插值模式 ('max_pool', 'nearest', 'bilinear')
        """
        if self.resize_shape is None: return batch_tensor
        T, C, H, W = batch_tensor.shape[1:]
        target_H, target_W = self.resize_shape
        if H == target_H and W == target_W: return batch_tensor
        
        is_bool = batch_tensor.dtype == torch.bool
        if is_bool: batch_tensor = batch_tensor.float()
        
        B = batch_tensor.shape[0]
        batch_tensor = batch_tensor.view(B * T, C, H, W)
        
        if mode == 'max_pool':
            # 降采样使用 Adaptive Max Pool 保留极值
            if target_H < H or target_W < W:
                processed_tensor = F.adaptive_max_pool2d(batch_tensor, output_size=self.resize_shape)
            else:
                processed_tensor = F.interpolate(batch_tensor, size=self.resize_shape, mode='bilinear', align_corners=False)
        elif mode in ['nearest', 'bilinear']:
            align = False if mode == 'bilinear' else None
            processed_tensor = F.interpolate(batch_tensor, size=self.resize_shape, mode=mode, align_corners=align)
        else:
            raise ValueError(f"Unsupported interpolation mode: {mode}")

        processed_tensor = processed_tensor.view(B, T, C, target_H, target_W)
        if is_bool: processed_tensor = processed_tensor.bool()
        return processed_tensor

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """配置优化器和学习率调度器"""
        max_epochs = getattr(self.hparams, 'max_epochs', 30)
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
        """前向传播，输出 Logits [B, T, Num_Bins, H, W]"""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """训练步"""
        _, x, y, target_mask, _ = batch
        target_mask = target_mask.bool()

        # 数据插值对齐
        x = self._interpolate_batch_gpu(x, mode='max_pool')
        y = self._interpolate_batch_gpu(y, mode='max_pool')
        target_mask = self._interpolate_batch_gpu(target_mask, mode='nearest')
        
        # 还原物理数值 (mm)
        MM_MAX_PHYSICAL = 30.0 
        y_for_binning = y * MM_MAX_PHYSICAL 

        # 前向计算 Logits
        logits_pred = self(x)
        
        # 计算 Loss (包含 Soft Label 和 Focal 机制)
        loss = self.criterion(logits_pred, y_for_binning, mask=target_mask)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def _decode_prediction(self, logits: torch.Tensor, method='expectation', temperature=0.8) -> torch.Tensor:
        """ 
        解码预测结果 (Logits -> Normalized Value [0,1])
        
        Args:
            logits: 模型输出 [B, T, Num_Bins, H, W]
            method: 解码方法 ('expectation' 或 'argmax')
            temperature: 温度系数。T < 1.0 (如 0.8) 会锐化分布，增强置信度；T > 1.0 会平滑分布。
                         在推理和验证时，推荐使用 0.8 以获得更锐利的预测。
        """
        # 1. 应用温度缩放 (Temperature Scaling)
        logits = logits / temperature
        
        # 2. 计算 Softmax 概率分布
        probs = F.softmax(logits, dim=2) # [B, T, Num_Bins, H, W]
        
        # 确保 bin_tool 在正确设备
        if hasattr(self.bin_tool, 'to') and self.bin_tool.device != logits.device:
            self.bin_tool.to(logits.device)
            
        centers = self.bin_tool.centers.to(logits.device) # [Num_Bins]
        
        if method == 'expectation':
            # === 期望解码 (Soft Argmax) ===
            # y = sum(p_i * center_i)
            # 调整 centers 形状以进行广播: [1, 1, Num_Bins, 1, 1]
            centers_reshaped = centers.view(1, 1, -1, 1, 1)
            y_pred = (probs * centers_reshaped).sum(dim=2) # -> [B, T, H, W]
        else:
            # === Argmax 解码 ===
            pred_idx = torch.argmax(probs, dim=2)
            y_pred = self.bin_tool.class_to_value(pred_idx) 
        
        # 阈值清理 (去除极小的底噪)
        y_pred[y_pred < 0.05] = 0.0
        
        # 归一化回 [0, 1] 区间
        MM_MAX = 30.0
        y_pred_normalized = y_pred / MM_MAX
        
        # 恢复 Channel 维度 -> [B, T, 1, H, W]
        return torch.clamp(y_pred_normalized, 0.0, 1.0).unsqueeze(2)

    def validation_step(self, batch, batch_idx):
        """验证步"""
        metadata, x, y, target_mask, input_mask = batch
        x = self._interpolate_batch_gpu(x, mode='max_pool')
        y = self._interpolate_batch_gpu(y, mode='max_pool')
        target_mask = self._interpolate_batch_gpu(target_mask, mode='nearest')
        
        logits_pred = self(x)
        
        MM_MAX_PHYSICAL = 30.0 
        y_for_binning = y * MM_MAX_PHYSICAL

        # 1. 计算 Val Loss
        val_loss = self.criterion(logits_pred, y_for_binning, mask=target_mask)
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, sync_dist=True)

        # 2. 解码预测 (使用 Expectation + Temperature=0.8)
        y_pred_clamped = self._decode_prediction(logits_pred, method='expectation', temperature=0.8)
        
        # 3. 计算 MAE
        val_mae = F.l1_loss(y_pred_clamped, y)
        self.log('val_mae', val_mae, on_epoch=True, sync_dist=True)

        # 4. 计算 TS 评分 (Metric)
        pred_mm = y_pred_clamped.squeeze(2) * MM_MAX_PHYSICAL
        target_mm = y.squeeze(2) * MM_MAX_PHYSICAL
        
        # 阈值与权重 (复用竞赛规则)
        thresholds = [0.1, 1.0, 2.0, 5.0, 8.0]
        level_weights = [0.1, 0.1, 0.2, 0.25, 0.35]
        
        # 时效权重
        time_weights_list = [
            0.0075, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
            0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.0075, 0.005
        ]
        T_out = pred_mm.shape[1]
        if T_out == 20:
            time_weights = torch.tensor(time_weights_list, device=self.device)
        else:
            time_weights = torch.ones(T_out, device=self.device) / T_out

        total_score = 0.0
        total_level_weight = sum(level_weights)

        for t_val, w_level in zip(thresholds, level_weights):
            hits = ((pred_mm >= t_val) & (target_mm >= t_val)).float().sum(dim=(0, 2, 3))
            misses = ((pred_mm < t_val) & (target_mm >= t_val)).float().sum(dim=(0, 2, 3))
            false_alarms = ((pred_mm >= t_val) & (target_mm < t_val)).float().sum(dim=(0, 2, 3))
            
            ts_t = hits / (hits + misses + false_alarms + 1e-6)
            ts_weighted_time = (ts_t * time_weights).sum()
            total_score += ts_weighted_time * w_level

        val_score = total_score / total_level_weight
        self.log('val_score', val_score, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        """测试步"""
        metadata, x, y, target_mask, input_mask = batch
        x = self._interpolate_batch_gpu(x, mode='max_pool')
        y = self._interpolate_batch_gpu(y, mode='max_pool')
        target_mask = self._interpolate_batch_gpu(target_mask, mode='nearest')
        
        logits_pred = self(x)
        
        # 测试时使用与验证一致的解码策略
        y_pred_clamped = self._decode_prediction(logits_pred, method='expectation', temperature=0.8)
        
        MM_MAX_PHYSICAL = 30.0 
        y_for_binning = y * MM_MAX_PHYSICAL
        test_loss = self.criterion(logits_pred, y_for_binning, mask=target_mask)
        self.log('test_loss', test_loss, on_epoch=True)
        
        return {
            'inputs': x[0].cpu().float().numpy(),
            'preds': y_pred_clamped[0].cpu().float().numpy(),
            'trues': y[0].cpu().float().numpy()
        }

    def infer_step(self, batch, batch_idx):
        """推理步"""
        metadata, x, input_mask = batch 
        x = self._interpolate_batch_gpu(x, mode='max_pool')
        logits_pred = self(x)
        # 推理时使用 expectation + temperature=0.8，获得高质量预测
        y_pred_clamped = self._decode_prediction(logits_pred, method='expectation', temperature=0.8)
        return y_pred_clamped

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.infer_step(batch, batch_idx)