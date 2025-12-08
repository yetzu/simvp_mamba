# metai/model/simvp/simvp_trainer.py

import subprocess
import os
import sys
import time
import glob
from typing import Any, cast, Dict, Optional, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as l
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from metai.model.core import get_optim_scheduler, timm_schedulers
from .simvp_model import SimVP_Model
from .simvp_loss import HybridLoss

class SimVP(l.LightningModule):
    def __init__(self, **args):
        super(SimVP, self).__init__()
        
        self.save_hyperparameters()
        config: Dict[str, Any] = dict(args)
        
        # 1. 模型初始化
        self.model = self._build_model(config)
        
        # 2. Loss 配置 (初始化值会被 Curriculum 覆盖，但仍需定义)
        loss_weight_l1 = config.get('loss_weight_l1', 1.0)
        loss_weight_ssim = config.get('loss_weight_ssim', 0.5)
        loss_weight_csi = config.get('loss_weight_csi', 1.0)
        loss_weight_spectral = config.get('loss_weight_spectral', 0.1)
        loss_weight_evo = config.get('loss_weight_evo', 0.5)

        self.criterion = HybridLoss(
            l1_weight=loss_weight_l1,
            ssim_weight=loss_weight_ssim,
            csi_weight=loss_weight_csi,
            spectral_weight=loss_weight_spectral,
            evo_weight=loss_weight_evo
        )
        
        rs = config.get('resize_shape', None)
        self.resize_shape = tuple(rs) if rs is not None else None

        # 课程学习配置
        self.use_curriculum_learning = config.get('use_curriculum_learning', True)
        
        # 测试相关配置
        self.auto_test_after_epoch = config.get('auto_test_after_epoch', True)
        self.test_script_path = config.get('test_script_path', None)
        if self.test_script_path is None:
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            script_path = os.path.join(current_dir, 'run.scwds.simvp.sh')
            if os.path.exists(script_path):
                self.test_script_path = script_path
            else:
                self.test_script_path = 'run.scwds.simvp.sh'
    
    def _build_model(self, config: Dict[str, Any]):
        return SimVP_Model(
             in_shape=config.get('in_shape'), hid_S=config.get('hid_S', 128), 
             hid_T=config.get('hid_T', 512), N_S=config.get('N_S', 4), N_T=config.get('N_T', 12),
             model_type=config.get('model_type', 'mamba'), out_channels=config.get('out_channels', 1),
             mlp_ratio=config.get('mlp_ratio', 8.0), drop=config.get('drop', 0.0), drop_path=config.get('drop_path', 0.1),
             spatio_kernel_enc=config.get('spatio_kernel_enc', 3), 
             spatio_kernel_dec=config.get('spatio_kernel_dec', 3),
             aft_seq_length=config.get('aft_seq_length', 20)
        )
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        配置优化器。注意：Config 中的 min_lr 应设置为 1e-5，
        以确保在 Curriculum 的 Phase 3 (高 CSI 权重) 阶段，模型仍有足够的更新步长。
        """
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
    
    def on_train_epoch_start(self):
        """
        🚀 [Fast-Track] 激进型课程学习策略
        目标：在较少 Epoch 内快速提升竞赛 Score
        """
        if not self.use_curriculum_learning:
            return
        
        epoch = self.current_epoch
        max_epochs = getattr(self.hparams, 'max_epochs', 50) # 假设默认50轮
        
        # 归一化进度 (0.0 -> 1.0)
        progress = epoch / max_epochs
        
        # === 动态权重计算 ===
        
        # 1. L1 (基础约束): 快速下降
        # 从 10.0 快速降到 1.0，后期不再过分关注像素级平滑
        # 逻辑: 前期靠强 L1 快速成型，后期放手让 CSI 优化细节
        l1_w = 10.0 - (9.0 * (progress ** 0.5)) 
        l1_w = max(l1_w, 1.0) 

        # 2. SSIM (结构): 保持稳定
        ssim_w = 1.0 - 0.5 * progress

        # 3. CSI (核心提分项): 激进增长
        # 从 0.5 开始 (不再是0!)，指数增长到 5.0
        # 逻辑: 一开始就要关注阈值命中率
        csi_w = 0.5 + 4.5 * (progress ** 2)

        # 4. Spec & Evo (辅助): 缓慢增加
        spec_w = 0.1 * progress
        evo_w = 0.5 * progress

        weights = {'l1': l1_w, 'ssim': ssim_w, 'evo': evo_w, 'spec': spec_w, 'csi': csi_w}
        
        # 更新权重
        if hasattr(self, 'criterion') and hasattr(self.criterion, 'weights'):
            self.criterion.weights.update(weights)
        
        # 日志记录
        if self.trainer.is_global_zero:
            w_str = ", ".join([f"{k}={v:.4f}" for k, v in weights.items()])
            print(f"\n[Fast-Curriculum] Epoch {epoch}/{max_epochs} | Progress: {progress:.2f}")
            print(f"                  Weights: {w_str}")
        
        # TensorBoard
        for k, v in weights.items():
            self.log(f"train/weight_{k}", v, on_epoch=True, sync_dist=True)

#     def on_train_epoch_end(self):
#         """后台非阻塞式测试"""
#         if self.trainer.is_global_zero and self.auto_test_after_epoch:
#             try:
#                 if not self.test_script_path: return
#                 script_path = str(self.test_script_path)
#                 if not os.path.isabs(script_path):
#                     current_file = os.path.abspath(__file__)
#                     project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
#                     script_path = os.path.join(project_root, script_path)
                
#                 if not os.path.exists(script_path): return
                
#                 save_dir = None
#                 if hasattr(self, 'hparams'):
#                     save_dir = self.hparams.get('save_dir') if isinstance(self.hparams, dict) else getattr(self.hparams, 'save_dir', None)
#                 if save_dir is None: save_dir = getattr(self.trainer, 'default_root_dir', os.getcwd())

#                 script_dir = os.path.dirname(script_path) or os.getcwd()
#                 log_dir = os.path.join(script_dir, 'test_logs')
#                 os.makedirs(log_dir, exist_ok=True)
                
#                 epoch = self.current_epoch
#                 log_file = os.path.join(log_dir, f'test_epoch_{epoch:03d}.log')
                
#                 # 构造后台执行代码
#                 background_code = f"""
# import os, time, glob, subprocess, sys
# save_dir = r'{save_dir}'
# script_path = r'{script_path}'
# epoch = {epoch}
# max_wait = 600

# start_time = time.time()
# found = False
# while time.time() - start_time < max_wait:
#     files = glob.glob(os.path.join(save_dir, "*.ckpt"))
#     target = [f for f in files if f"epoch={{epoch:02d}}" in f]
#     if target:
#         size1 = os.path.getsize(target[0])
#         time.sleep(2)
#         if os.path.getsize(target[0]) == size1 and size1 > 0:
#             found = True
#             break
#     time.sleep(5)

# with open(r'{log_file}', 'w') as f:
#     if found:
#         f.write(f"[Background] Found checkpoint for Epoch {{epoch}}. Starting Test...\\n")
#         f.flush()
#         try:
#             subprocess.run(['bash', script_path, 'test'], stdout=f, stderr=subprocess.STDOUT, cwd=r'{script_dir}')
#         except Exception as e:
#             f.write(f"\\n[Background Error] {{e}}\\n")
#     else:
#         f.write(f"[Background] Timeout waiting for checkpoint. Test Skipped.\\n")
# """
#                 subprocess.Popen([sys.executable, '-c', background_code], cwd=script_dir, start_new_session=True)
                
#             except Exception as e:
#                 print(f"[ERROR] Failed to launch test script: {e}")
    
    def forward(self, x):
        return self.model(x)
    
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
    
    def training_step(self, batch, batch_idx):
        _, x, y, target_mask, _ = batch
        target_mask = target_mask.bool()

        x = self._interpolate_batch_gpu(x, mode='max_pool')
        y = self._interpolate_batch_gpu(y, mode='max_pool')
        target_mask = self._interpolate_batch_gpu(target_mask, mode='nearest')

        logits_pred = self(x)
        loss, loss_dict = self.criterion(logits_pred, y, mask=target_mask)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        for comp in ['l1', 'ssim', 'csi', 'spec', 'evo']:
            if comp in loss_dict:
                self.log(f'train_loss_{comp}', loss_dict[comp], on_step=True, on_epoch=True, prog_bar=False)
            if f'{comp}_weighted' in loss_dict:
                self.log(f'train_loss_{comp}_weighted', loss_dict[f'{comp}_weighted'], on_step=True, on_epoch=True, prog_bar=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        metadata, x, y, target_mask, input_mask = batch
        target_mask = target_mask.bool()

        # 1. 前向传播与插值
        x = self._interpolate_batch_gpu(x, mode='max_pool')
        y = self._interpolate_batch_gpu(y, mode='max_pool')
        target_mask = self._interpolate_batch_gpu(target_mask, mode='nearest')
        
        logits_pred = self(x)
        y_pred = torch.sigmoid(logits_pred)
        y_pred_clamped = torch.clamp(y_pred, 0.0, 1.0)
        
        # 2. 计算 Loss (保持不变)
        loss, loss_dict = self.criterion(logits_pred, y, mask=target_mask)
        for comp in ['l1', 'ssim', 'csi', 'spec', 'evo']:
            if comp in loss_dict:
                self.log(f'val_loss_{comp}', loss_dict[comp], on_epoch=True, sync_dist=True)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)

        # ====================================================
        # 3. [优化] 对齐官方规则的评分计算
        # ====================================================
        MM_MAX = 30.0
        pred_mm = y_pred_clamped * MM_MAX
        target_mm = y * MM_MAX

        # A. 官方阈值与强度权重 (Table 2)
        # 去掉了 0.01 (噪音)，对齐官方 0.1 起步
        thresholds = [0.1, 1.0, 2.0, 5.0, 8.0]
        level_weights = [0.1, 0.1, 0.2, 0.25, 0.35]
        
        # B. 官方时效权重 (Table 1) - 针对 20 帧
        # 对应 6min 到 120min
        time_weights_list = [
            0.0075, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,  # 1-10 (60min 权重最高)
            0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.0075, 0.005 # 11-20
        ]
        # 转换为 Tensor 并移动到对应设备
        T_out = pred_mm.shape[1]
        if T_out == 20:
            time_weights = torch.tensor(time_weights_list, device=self.device)
        else:
            # 如果输出不是20帧，则平均分配
            time_weights = torch.ones(T_out, device=self.device) / T_out

        # C. 计算加权 TS (Weighted TS)
        # 这种计算方式是 "Micro-average over Batch, but Macro over Time/Level"
        # 既保留了批量计算的速度，又引入了时效权重
        
        total_score = 0.0
        total_level_weight = sum(level_weights)

        for t_val, w_level in zip(thresholds, level_weights):
            # [B, T, H, W] (C=1, 已去除) -> Bool
            # 注意：target_mm 和 pred_mm 可能是 [B, T, 1, H, W]，需要squeeze或指定sum dim
            # 在training step中，logits是 [B, T, C, H, W]，squeeze后 [B, T, H, W]
            # 让我们兼容这两种情况
            
            # 确保是 [B, T, H, W]
            if pred_mm.dim() == 5 and pred_mm.shape[2] == 1:
                 p_mm = pred_mm.squeeze(2)
                 t_mm = target_mm.squeeze(2)
            else:
                 p_mm = pred_mm
                 t_mm = target_mm

            hits_tensor = (p_mm >= t_val) & (t_mm >= t_val)
            misses_tensor = (p_mm < t_val) & (t_mm >= t_val)
            false_alarms_tensor = (p_mm >= t_val) & (t_mm < t_val)
            
            # 在 [B, H, W] 维度求和，保留 [T] 维度以应用时效权重
            # sum dim: 0(Batch), 2(H), 3(W) -> Result shape: [T]
            # [FIX]: 输入已经是 4D [B, T, H, W]，所以 dim=(0, 2, 3) 是正确的
            # 如果输入是 5D [B, T, C, H, W]，则需要 dim=(0, 2, 3, 4)
            
            if p_mm.dim() == 4:
                sum_dims = (0, 2, 3)
            else: # 5D
                sum_dims = (0, 2, 3, 4)

            hits = hits_tensor.float().sum(dim=sum_dims)
            misses = misses_tensor.float().sum(dim=sum_dims)
            false_alarms = false_alarms_tensor.float().sum(dim=sum_dims)
            
            # 计算每帧的 TS: [T]
            ts_t = hits / (hits + misses + false_alarms + 1e-6)
            
            # 应用时效权重: sum( [T] * [T] ) -> Scalar
            ts_weighted_time = (ts_t * time_weights).sum()
            
            # 累加强度分级得分
            total_score += ts_weighted_time * w_level

        # 归一化 (虽然 level_weights 和为 1，但保持严谨)
        val_score = total_score / total_level_weight

        # 4. 记录指标
        self.log('val_score', val_score, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # 额外记录 MAE 供参考 (不参与 EarlyStopping，因为 MAE 容易被 0 值主导)
        val_mae = F.l1_loss(y_pred_clamped, y)
        self.log('val_mae', val_mae, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        metadata, x, y, target_mask, input_mask = batch
        target_mask = target_mask.bool()
        x = self._interpolate_batch_gpu(x, mode='max_pool')
        y = self._interpolate_batch_gpu(y, mode='max_pool')
        target_mask = self._interpolate_batch_gpu(target_mask, mode='nearest')

        logits_pred = self(x)
        y_pred = torch.sigmoid(logits_pred)
        y_pred_clamped = torch.clamp(y_pred, 0.0, 1.0)
        
        with torch.no_grad():
            loss, loss_dict = self.criterion(logits_pred, y, mask=target_mask)
            for comp in ['l1', 'ssim', 'csi', 'spec', 'evo']:
                if comp in loss_dict:
                    self.log(f'test_loss_{comp}', loss_dict[comp], on_epoch=True)
            
        self.log('test_loss', loss, on_epoch=True)
        
        return {
            'inputs': x[0].cpu().float().numpy(),
            'preds': y_pred_clamped[0].cpu().float().numpy(),
            'trues': y[0].cpu().float().numpy()
        }
    
    def infer_step(self, batch, batch_idx):
        metadata, x, input_mask = batch 
        x = self._interpolate_batch_gpu(x, mode='max_pool')
        logits_pred = self(x)
        y_pred = torch.sigmoid(logits_pred)
        return torch.clamp(y_pred, 0.0, 1.0)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.infer_step(batch, batch_idx)