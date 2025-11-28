# metai/model/simvp/simvp_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

# 尝试导入 torchmetrics，如果不存在则提供回退方案
try:
    from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("Warning: torchmetrics not found. MS-SSIM will be skipped.")


class WeightedScoreSoftCSILoss(nn.Module):
    """
    基于竞赛评分规则的 Soft-CSI 损失函数，支持强度加权、时效加权和 Masking。
    严格对齐比赛评分表的阈值和权重。
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.MM_MAX = 30.0 
        
        # --- 1. 对齐强度分级及权重 (表2) ---
        # 阈值: 0.1, 1.0, 2.0, 5.0, 8.0 (mm)
        thresholds_raw = [0.1, 1.0, 2.0, 5.0, 8.0]
        # 权重: 0.1, 0.1, 0.2, 0.25, 0.35
        weights_raw    = [0.1, 0.1, 0.2, 0.25, 0.35]
        
        self.register_buffer('thresholds', torch.tensor(thresholds_raw) / self.MM_MAX)
        self.register_buffer('intensity_weights', torch.tensor(weights_raw))
        
        # --- 2. 对齐时效及权重 (表1) ---
        time_weights_raw = [
            0.0075, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
            0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.0075, 0.005 
        ]
        self.register_buffer('time_weights', torch.tensor(time_weights_raw).view(1, -1, 1, 1))
        
        self.smooth = smooth

    def forward(self, pred, target, mask=None):
        """
        pred: [B, T, H, W], 范围 [0, 1]
        target: [B, T, H, W], 范围 [0, 1]
        mask: [B, T, H, W] 或 [B, 1, H, W]
        """
        T = pred.shape[1]
        current_time_weights = self.time_weights[:, :T, :, :]
        # 归一化时间权重
        current_time_weights = current_time_weights / current_time_weights.mean()
        
        # 统一 Mask 维度 (如果 mask 存在)
        if mask is not None:
            if mask.dim() == 4 and mask.shape[1] == 1 and pred.shape[1] > 1:
                mask = mask.expand(-1, pred.shape[1], -1, -1)
            elif mask.dim() == 5:
                mask = mask.squeeze(2)

        total_weighted_loss = 0.0
        total_weight_sum = 0.0

        for i, t in enumerate(self.thresholds):
            w = self.intensity_weights[i]
            
            # 1. 软二值化 (Sigmoid temp=50 模拟阶跃)
            pred_score = torch.sigmoid((pred - t) * 50)
            target_score = (target > t).float()
            
            # 2. 应用 Mask
            if mask is not None:
                pred_score = pred_score * mask
                target_score = target_score * mask
                
            # 3. 计算 Intersection (TP) 和 Union (TP + FN + FP)
            # 在空间维度 (H, W) 求和
            intersection = (pred_score * target_score).sum(dim=(-2, -1))
            total_pred = pred_score.sum(dim=(-2, -1))
            total_target = target_score.sum(dim=(-2, -1))
            union = total_pred + total_target - intersection
            
            # 4. 计算 CSI
            csi = (intersection + self.smooth) / (union + self.smooth)
            loss_map = 1.0 - csi
            
            # 5. 应用时间权重 (在时间维度 T 平均)
            weighted_loss_t = (loss_map * current_time_weights.squeeze(-1).squeeze(-1)).mean()
            
            # 6. 应用强度权重
            total_weighted_loss += weighted_loss_t * w
            total_weight_sum += w

        return total_weighted_loss / total_weight_sum


class LogSpectralDistanceLoss(nn.Module):
    """
    频域损失。用于抗模糊，强制模型在频域保持高频分量。
    """
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target, mask=None): 
        # FFT 变换需要 float32
        pred_fp32 = pred.float()
        target_fp32 = target.float()
        
        # FFT 变换 (实数输入，复数输出)
        pred_fft = torch.fft.rfft2(pred_fp32, dim=(-2, -1), norm='ortho')
        target_fft = torch.fft.rfft2(target_fp32, dim=(-2, -1), norm='ortho')
        
        # 幅度谱
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # 对数距离 (L1 loss on log-magnitude)
        loss = F.l1_loss(torch.log(pred_mag + self.epsilon), torch.log(target_mag + self.epsilon))
        
        return loss


class WeightedEvolutionLoss(nn.Module):
    """
    物理感知的加权演变损失。对强回波区域的时间变化赋予更高权重。
    """
    def __init__(self, weight_scale=5.0):
        super().__init__()
        self.weight_scale = weight_scale

    def forward(self, pred, target, mask=None):
        # 计算时间差分 (dI/dt)
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        
        # 计算误差
        diff_error = torch.abs(pred_diff - target_diff)
        
        # 动态加权：如果该位置是强回波，则赋予更高权重
        weight_map = 1.0 + self.weight_scale * target[:, 1:]
        
        # 应用 Mask
        if mask is not None:
            if mask.dim() == 5:
                mask = mask.squeeze(2)
            
            # 取 T-1 帧的 Mask (代表 t+1 时刻的有效性)
            mask_t_plus_1 = mask[:, 1:] 
            
            diff_error = diff_error * mask_t_plus_1 
            weight_map = weight_map * mask_t_plus_1 
            
            count_valid = mask_t_plus_1.sum()
            if count_valid > 0:
                weighted_loss = (diff_error * weight_map).sum() / count_valid
            else:
                weighted_loss = 0.0 
        else:
            weighted_loss = (diff_error * weight_map).mean()

        return weighted_loss


class HybridLoss(nn.Module):
    """
    Mamba 物理感知混合损失函数 (SOTA 优化版)
    """
    def __init__(self, 
                 l1_weight=1.0, 
                 ssim_weight=0.5, 
                 csi_weight=1.0, 
                 spectral_weight=0.1, 
                 evo_weight=0.5):
        super().__init__()
        self.weights = {
            'l1': l1_weight,
            'ssim': ssim_weight,
            'csi': csi_weight,
            'spec': spectral_weight,
            'evo': evo_weight
        }
        
        # [关键] 必须使用 reduction='none' 才能支持后续的 Pixel-Wise 加权和 Masking
        self.l1 = nn.L1Loss(reduction='none') 
        
        if TORCHMETRICS_AVAILABLE and ssim_weight > 0:
            self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0, reduction='none')
        else:
            self.ms_ssim = None
            
        self.soft_csi = WeightedScoreSoftCSILoss()
        self.spectral = LogSpectralDistanceLoss()
        self.evolution = WeightedEvolutionLoss()

    def forward(self, logits, target, mask=None):
        """
        logits: [B, T, C, H, W] - 模型的原始输出 (Raw Logits)
        target: [B, T, C, H, W] - 归一化后的真实值 [0, 1]
        mask: [B, T, C, H, W] 或 [B, T, H, W] - 0/1 张量
        """
        # 1. 预处理
        if logits.dim() == 5: logits = logits.squeeze(2)
        if target.dim() == 5: target = target.squeeze(2)
        if mask is not None and mask.dim() == 5: mask = mask.squeeze(2)
        
        # 将 logits 转为 [0, 1] 概率
        pred = torch.sigmoid(logits)
        
        loss_dict = {}
        total_loss = 0.0
        
        # =====================================================================
        # 2. L1 Loss (Pixel-Wise) - 增加"难例挖掘" (Hard Example Mining)
        # =====================================================================
        # 计算基础 L1 误差
        l1_loss_map = self.l1(pred, target) # [B, T, H, W]
        
        # [关键修正] 动态权重：基于赛题评分表对强降水区域进行惩罚加倍
        # 评分表关键阈值：2.0mm, 5.0mm, 8.0mm
        # 归一化基准: MM_MAX = 30.0
        
        pixel_weight = torch.ones_like(target)
        
        # Level 1: > 2.0mm (权重 0.1 -> 0.2) -> 设为 x2 关注度
        # 2.0 / 30.0 = 0.0667
        pixel_weight[target > (2.0 / 30.0)] = 2.0
        
        # Level 2: > 5.0mm (权重 0.2 -> 0.25) -> 设为 x5 关注度
        # 5.0 / 30.0 = 0.1667
        pixel_weight[target > (5.0 / 30.0)] = 5.0
        
        # Level 3: > 8.0mm (权重 0.25 -> 0.35, 最高分) -> 设为 x20 关注度 (决胜点)
        # 8.0 / 30.0 = 0.2667
        pixel_weight[target > (8.0 / 30.0)] = 20.0
        
        # 应用动态权重
        l1_loss_map = l1_loss_map * pixel_weight
        
        # 应用有效区域 Mask
        if mask is not None:
            masked_error = l1_loss_map * mask
            count_valid = mask.sum()
            # 避免除以 0
            l1_loss = masked_error.sum() / (count_valid + 1e-8)
        else:
            l1_loss = l1_loss_map.mean()
            
        total_loss += self.weights['l1'] * l1_loss
        loss_dict['l1'] = l1_loss.item() if isinstance(l1_loss, torch.Tensor) else l1_loss
        # =====================================================================
        
        # 3. Soft-CSI Loss (直接优化评价指标)
        if self.weights['csi'] > 0:
            csi_loss = self.soft_csi(pred, target, mask)
            total_loss += self.weights['csi'] * csi_loss
            loss_dict['csi'] = csi_loss.item()
            
        # 4. Spectral Loss (频域抗模糊)
        if self.weights['spec'] > 0:
            spec_loss = self.spectral(pred, target, mask)
            total_loss += self.weights['spec'] * spec_loss
            loss_dict['spec'] = spec_loss.item()
            
        # 5. Evolution Loss (时序演变约束)
        if self.weights['evo'] > 0 and pred.shape[1] > 1:
            evo_loss = self.evolution(pred, target, mask)
            total_loss += self.weights['evo'] * evo_loss
            loss_dict['evo'] = evo_loss.item()
            
        # 6. MS-SSIM Loss (结构一致性)
        if self.ms_ssim is not None and self.weights['ssim'] > 0:
            pred_c = pred.view(-1, 1, pred.shape[-2], pred.shape[-1])
            target_c = target.view(-1, 1, target.shape[-2], target.shape[-1])
            
            if mask is not None:
                mask_c = mask.view(-1, 1, mask.shape[-2], mask.shape[-1])
                pred_c = pred_c * mask_c
                target_c = target_c * mask_c
            
            ssim_val = self.ms_ssim(pred_c, target_c).mean()
            ssim_loss = 1.0 - ssim_val
            total_loss += self.weights['ssim'] * ssim_loss
            loss_dict['ssim'] = ssim_loss.item()
        
        # 记录加权后的总 Loss
        loss_dict['total'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        
        return total_loss, loss_dict