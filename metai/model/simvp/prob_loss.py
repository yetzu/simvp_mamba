import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ProbabilisticBinningTool:
    def __init__(self, num_bins=40, max_val=30.0, device='cpu'): # 默认 num_bins 改为 40
        self.num_bins = num_bins
        self.max_val = max_val
        self.device = device
        
        # 优化分箱边缘：0-1mm 细分，1-8mm 过渡，8-30mm 覆盖高值
        # 40 bins 策略
        edges_1 = np.linspace(0.0, 1.0, 11) 
        edges_2 = np.linspace(1.0, 8.0, 15) # 减少中间冗余
        edges_3 = np.linspace(8.0, max_val, num_bins - 25 + 1)
        
        all_edges = np.concatenate([edges_1, edges_2, edges_3])
        self.edges_np = np.unique(all_edges)
        
        if len(self.edges_np) > num_bins + 1:
             self.edges_np = self.edges_np[:num_bins+1]
        
        self.num_classes = len(self.edges_np) - 1
        self.centers_np = (self.edges_np[:-1] + self.edges_np[1:]) / 2.0
        self.centers_np[0] = 0.0 # 修正第一个 center 为 0
        
        # 计算基础权重（用于初始化平滑权重）
        self.class_weights_np = np.ones(self.num_classes, dtype=np.float32)
        idx_8  = np.searchsorted(self.edges_np, 8.0, side='right') - 1
        self.class_weights_np[idx_8:] = 2.0 # 初始高值区略高

        self.edges = torch.tensor(self.edges_np, dtype=torch.float32).to(device)
        self.centers = torch.tensor(self.centers_np, dtype=torch.float32).to(device)
        self.class_weights = torch.tensor(self.class_weights_np, dtype=torch.float32).to(device)

    def to(self, device):
        self.device = device
        self.edges = self.edges.to(device)
        self.centers = self.centers.to(device)
        self.class_weights = self.class_weights.to(device)
        return self

    def to_class(self, value_tensor):
        edges_on_device = self.edges.to(value_tensor.device)
        idxs = torch.bucketize(value_tensor, edges_on_device) - 1
        return torch.clamp(idxs, 0, self.num_classes - 1)

    def class_to_value(self, class_idxs):
        return self.centers.to(class_idxs.device)[class_idxs]

class ProbabilisticCrossEntropyLoss(nn.Module):
    """
    Gaussian Soft-Label Focal Loss
    """
    def __init__(self, num_bins=40, max_val=30.0, sigma=2.0, use_focal=True, gamma=2.0):
        super().__init__()
        self.num_bins = num_bins
        self.max_val = max_val
        self.sigma = sigma
        self.use_focal = use_focal
        self.gamma = gamma
        
        self.bin_tool = ProbabilisticBinningTool(num_bins, max_val, device='cpu')
        
        # 权重平滑策略
        raw_weights = self.bin_tool.class_weights_np
        # Log平滑，避免权重过大
        smoothed_weights = np.log1p(raw_weights) + 1.0 
        self.register_buffer('smooth_weights', torch.tensor(smoothed_weights, dtype=torch.float32))

    def forward(self, logits, target, mask=None):
        if self.bin_tool.device != logits.device:
            self.bin_tool.to(logits.device)
        
        # 1. 构建高斯软标签
        with torch.no_grad():
            target_squeeze = target.squeeze(2)
            target_cls = self.bin_tool.to_class(target_squeeze)
            
            bin_indices = torch.arange(self.num_bins, device=logits.device).view(1, 1, 1, 1, -1)
            target_cls_unsqueezed = target_cls.unsqueeze(-1)
            
            # Gaussian Kernel
            dist_sq = (bin_indices - target_cls_unsqueezed).pow(2)
            soft_target = torch.exp(-dist_sq / (2 * self.sigma ** 2))
            soft_target = soft_target / (soft_target.sum(dim=-1, keepdim=True) + 1e-8)
            
            soft_target_flat = soft_target.view(-1, self.num_bins)
            target_flat = target_cls.reshape(-1)
            sample_weights = self.smooth_weights.to(logits.device)[target_flat]

        # 2. 计算 Loss
        B, T, C, H, W = logits.shape
        logits_flat = logits.permute(0, 1, 3, 4, 2).reshape(-1, C)
        log_probs = F.log_softmax(logits_flat, dim=1)
        
        # Soft Cross Entropy
        ce_loss = - (soft_target_flat * log_probs).sum(dim=1)
        
        # Focal Modulation
        if self.use_focal:
            # 这里的 p 取目标类别的概率近似，或者使用 soft_target 加权的概率
            # 简化版 Focal: (1 - p_target)^gamma * CE
            # p_target 近似为 exp(log_probs) 在 soft_target 下的期望
            probs = torch.exp(log_probs)
            p_target = (probs * soft_target_flat).sum(dim=1)
            focal_weight = (1 - p_target).pow(self.gamma)
            loss = focal_weight * ce_loss
        else:
            loss = ce_loss
            
        # Class Balance & Mask
        loss = loss * sample_weights
        
        if mask is not None:
            mask_flat = mask.view(-1)
            loss = (loss * mask_flat).sum() / (mask_flat.sum() + 1e-6)
        else:
            loss = loss.mean()
            
        return loss