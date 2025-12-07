import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ProbabilisticBinningTool:
    """
    [SOTA Optimized] 概率分箱工具类
    策略：5段式非均匀分箱 + 零值精确锁定 + 高值区高密度覆盖
    修复：严格对齐 num_bins 数量，消除 Off-by-one 误差
    """
    def __init__(self, num_bins=40, max_val=30.0, device='cpu'):
        self.num_bins = num_bins
        self.max_val = max_val
        self.device = device
        
        # --- 1. 低值区 [0.0 - 1.0] (权重 0.1) ---
        # 关键点：0.1 是有雨无雨的分界线，必须精确
        # Bins: [0.0, 0.1], [0.1, 0.4], [0.4, 0.7], [0.7, 1.0] -> 4 bins
        edges_1 = np.array([0.0, 0.1, 0.4, 0.7, 1.0])

        # --- 2. 过渡区 [1.0 - 2.0] (权重 0.1) ---
        # Bins: [1.0, 1.5], [1.5, 2.0] -> 2 bins
        edges_2 = np.linspace(1.5, 2.0, 2)

        # --- 3. 中值区 [2.0 - 5.0] (权重 0.2) ---
        # Bins: 2.0-2.5, ..., 4.5-5.0 -> 6 bins
        edges_3 = np.linspace(2.5, 5.0, 6)

        # --- 4. 强降水区 [5.0 - 8.0] (权重 0.25) ---
        # Bins: 5.0-5.5, ..., 7.5-8.0 -> 6 bins
        edges_4 = np.linspace(5.5, 8.0, 6)

        # --- 5. 极值决胜区 [8.0 - max] (权重 0.35) ---
        # 策略：将剩余算力全部投入此处。因为 >8.0mm 区域的 MAE 惩罚最重。
        # 已用: 4 + 2 + 6 + 6 = 18 bins
        used_bins = 18 
        remaining_bins = num_bins - used_bins
        
        if remaining_bins < 1:
            raise ValueError(f"num_bins ({num_bins}) is too small. Min required: 19.")

        # [Fix: Off-by-one] 使用 linspace 生成 exact 数量的点
        step = (max_val - 8.0) / remaining_bins
        edges_5 = np.linspace(8.0 + step, max_val, remaining_bins)
        
        # 合并与去重
        all_edges = np.concatenate([edges_1, edges_2, edges_3, edges_4, edges_5])
        self.edges_np = np.unique(all_edges)
        
        # 最终校验
        self.num_classes = len(self.edges_np) - 1
        if self.num_classes != num_bins:
            print(f"[CRITICAL WARNING] BinningTool: Generated {self.num_classes} bins, expected {num_bins}!")

        # 计算中心点 (解码用)
        self.centers_np = (self.edges_np[:-1] + self.edges_np[1:]) / 2.0
        self.centers_np[0] = 0.0 # [重要] Bin 0 强制对齐 0.0mm
        
        # 类别权重：对 >8.0mm (Index >= 18) 给予 2倍 关注
        self.class_weights_np = np.ones(self.num_classes, dtype=np.float32)
        if self.num_classes > 18:
            high_range_len = self.num_classes - 18
            boost_weights = np.linspace(10.0, 50.0, high_range_len)
            self.class_weights_np[18:] = boost_weights

        # 转 Tensor
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
        edges = self.edges.to(value_tensor.device)
        idxs = torch.bucketize(value_tensor, edges) - 1
        return torch.clamp(idxs, 0, self.num_classes - 1)

    def class_to_value(self, class_idxs):
        return self.centers.to(class_idxs.device)[class_idxs]

class ProbabilisticCrossEntropyLoss(nn.Module):
    """
    Gaussian Soft-Label Focal Loss (Zero-Aware / Hybrid)
    特性：针对 0 值点采用 Hard Label，针对有雨点采用 Soft Label。
    彻底解决"零值弥散"导致的底噪问题。
    """
    def __init__(self, num_bins=40, max_val=30.0, sigma=2.0, use_focal=True, gamma=2.0):
        super().__init__()
        self.num_bins = num_bins
        self.max_val = max_val
        self.sigma = sigma
        self.use_focal = use_focal
        self.gamma = gamma
        
        self.bin_tool = ProbabilisticBinningTool(num_bins, max_val, device='cpu')
        
        # 权重平滑
        raw_weights = self.bin_tool.class_weights_np
        smoothed_weights = np.log1p(raw_weights) + 1.0 
        self.register_buffer('smooth_weights', torch.tensor(smoothed_weights, dtype=torch.float32))

    def forward(self, logits, target, mask=None):
        """
        logits: [B, T, Num_Bins, H, W]
        target: [B, T, H, W] (物理数值)
        """
        if self.bin_tool.device != logits.device:
            self.bin_tool.to(logits.device)
        
        # 维度对齐与防御
        if target.dim() == 5: target = target.squeeze(2)
        if mask is not None and mask.dim() == 5: mask = mask.squeeze(2)

        # 1. 生成目标分布 (Target Distribution)
        with torch.no_grad():
            # 物理值 -> 类别索引
            target_cls = self.bin_tool.to_class(target) # [B, T, H, W]
            
            # 准备网格 [1, 1, 1, 1, Num_Bins]
            bin_idx = torch.arange(self.num_bins, device=logits.device).view(1, 1, 1, 1, -1)
            target_cls_unsqueeze = target_cls.unsqueeze(-1) # [B, T, H, W, 1]
            
            # --- 核心修复：混合标签生成 (Hybrid Label Generation) ---
            
            # A. 计算高斯分布 (Soft)
            dist_sq = (bin_idx - target_cls_unsqueeze).pow(2)
            soft_target = torch.exp(-dist_sq / (2 * self.sigma ** 2))
            
            # B. 强制处理零值 (Hard Zero)
            # 使用极小值阈值判断零值，比 ==0 更健壮
            is_zero = (target.unsqueeze(-1) < 1e-4)
            
            # 构造 Hard Label: 只有 index=0 处为 1.0，其余为 0
            hard_target = torch.zeros_like(soft_target)
            # 注意：bin 0 对应无雨
            hard_target[..., 0] = 1.0 
            
            # 混合：无雨处用 Hard，有雨处用 Soft
            # 这一步是消除"满图灰底"的关键
            final_target = torch.where(is_zero, hard_target, soft_target)
            
            # 归一化 (Sum=1)
            final_target = final_target / (final_target.sum(dim=-1, keepdim=True) + 1e-8)
            
            # 展平准备计算
            target_dist_flat = final_target.view(-1, self.num_bins)
            target_cls_flat = target_cls.reshape(-1)
            weights_flat = self.smooth_weights[target_cls_flat]

        # 2. 计算 Loss
        # Logits: [B, T, C, H, W] -> [B, T, H, W, C] -> [N, C]
        B, T, C, H, W = logits.shape
        logits_flat = logits.permute(0, 1, 3, 4, 2).reshape(-1, C)
        
        # 使用 fp32 进行 LogSoftmax 防止半精度溢出
        log_probs = F.log_softmax(logits_flat.float(), dim=1)
        
        # Cross Entropy
        ce = - (target_dist_flat * log_probs).sum(dim=1)
        
        # Focal Modulation
        if self.use_focal:
            pt = torch.exp(log_probs)
            # 目标概率 = sum(pred_prob * target_dist)
            pt_target = (pt * target_dist_flat).sum(dim=1)
            focal_w = (1 - pt_target).pow(self.gamma)
            loss = focal_w * ce
        else:
            loss = ce
            
        # 类平衡加权
        loss = loss * weights_flat
        
        # 3. Mask 聚合
        if mask is not None:
            mask_flat = mask.reshape(-1)
            valid = mask_flat.sum()
            if valid > 0:
                loss = (loss * mask_flat).sum() / valid
            else:
                loss = loss.sum() * 0.0
        else:
            loss = loss.mean()
            
        return loss