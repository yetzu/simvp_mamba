import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ProbabilisticBinningTool:
    """
    概率分箱工具类
    负责将连续的物理数值（降水强度）映射为离散的类别索引，并管理分箱的边界与中心值。
    
    [优化策略]
    基于竞赛评分规则表（Table 2），采用 5 段式非均匀分箱策略：
    1. [0.0 - 1.0]: 权重 0.1，分配 4 个 bins，确保包含 0.1 阈值。
    2. [1.0 - 2.0]: 权重 0.1，分配 2 个 bins。
    3. [2.0 - 5.0]: 权重 0.2，分配 6 个 bins (步长 0.5)。
    4. [5.0 - 8.0]: 权重 0.25，分配 6 个 bins (步长 0.5)。
    5. [8.0 - max]: 权重 0.35 (最高)，分配剩余所有 bins (约 22 个)，以最小化高值区的回归误差。
    """
    def __init__(self, num_bins=40, max_val=30.0, device='cpu'):
        self.num_bins = num_bins
        self.max_val = max_val
        self.device = device
        
        # --- 1. 区间 [0.0, 1.0] (权重 0.1) ---
        # 重点是分清 0.1 (有雨/无雨)
        # Bins: [0.0, 0.1], [0.1, 0.4], [0.4, 0.7], [0.7, 1.0] -> 4 bins
        edges_1 = np.array([0.0, 0.1, 0.4, 0.7, 1.0])

        # --- 2. 区间 [1.0, 2.0] (权重 0.1) ---
        # Bins: [1.0, 1.5], [1.5, 2.0] -> 2 bins
        # linspace(1.5, 2.0, 2) 生成 [1.5, 2.0]
        edges_2 = np.linspace(1.5, 2.0, 2)

        # --- 3. 区间 [2.0, 5.0] (权重 0.2) ---
        # Bins: 2.0-2.5, ..., 4.5-5.0 -> 6 bins
        # linspace(2.5, 5.0, 6) 生成 [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        edges_3 = np.linspace(2.5, 5.0, 6)

        # --- 4. 区间 [5.0, 8.0] (权重 0.25) ---
        # Bins: 5.0-5.5, ..., 7.5-8.0 -> 6 bins
        edges_4 = np.linspace(5.5, 8.0, 6)

        # --- 5. 区间 [8.0, max_val] (权重 0.35 - 决胜区) ---
        # 计算剩余可用的 bins 数量
        # 已用: 4 + 2 + 6 + 6 = 18 bins
        used_bins = 18
        remaining_bins = num_bins - used_bins
        
        if remaining_bins < 1:
            raise ValueError(f"num_bins ({num_bins}) is too small for the 5-segment strategy (min 19).")

        # 生成剩余区间的边缘点
        # 起始点为 8.0 + step，以避免与 8.0 重复
        step = (max_val - 8.0) / remaining_bins
        edges_5 = np.linspace(8.0 + step, max_val, remaining_bins)
        
        # 合并所有边缘
        all_edges = np.concatenate([edges_1, edges_2, edges_3, edges_4, edges_5])
        
        # 确保唯一性和排序 (虽然上述构造逻辑已保证，但作为防御性编程)
        self.edges_np = np.unique(all_edges)
        
        # 最终校验
        self.num_classes = len(self.edges_np) - 1
        if self.num_classes != num_bins:
            print(f"[Warning] ProbabilisticBinningTool: Generated {self.num_classes} bins, expected {num_bins}.")

        # 计算分箱中心点 (用于解码)
        self.centers_np = (self.edges_np[:-1] + self.edges_np[1:]) / 2.0
        # 修正：第一个 bin [0.0, 0.1] 的中心倾向于 0.0，代表无雨
        self.centers_np[0] = 0.0 
        
        # --- 计算类别权重 (Class Weights) ---
        # 根据 5 段式策略，对高值区 (>8.0mm) 进行加权
        # 前 18 个 bins 对应 < 8.0mm
        self.class_weights_np = np.ones(self.num_classes, dtype=np.float32)
        # 索引 18 及之后对应 >= 8.0mm 区域 (权重 0.35 vs 0.1/0.2)
        # 给予 2.0 倍的基础权重强化关注
        if self.num_classes > 18:
            self.class_weights_np[18:] = 2.0 

        # 转为 Tensor
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
        """将连续数值转换为类别索引"""
        edges_on_device = self.edges.to(value_tensor.device)
        # bucketize: left inclusive, right exclusive (indices 1..len-1)
        # 减 1 使索引从 0 开始
        idxs = torch.bucketize(value_tensor, edges_on_device) - 1
        # 截断越界值
        return torch.clamp(idxs, 0, self.num_classes - 1)

    def class_to_value(self, class_idxs):
        """将类别索引转换为中心值"""
        return self.centers.to(class_idxs.device)[class_idxs]

class ProbabilisticCrossEntropyLoss(nn.Module):
    """
    Gaussian Soft-Label Focal Loss
    结合了高斯软标签 (解决边界模糊) 和 Focal Loss (解决样本不平衡)。
    """
    def __init__(self, num_bins=40, max_val=30.0, sigma=2.0, use_focal=True, gamma=2.0):
        super().__init__()
        self.num_bins = num_bins
        self.max_val = max_val
        self.sigma = sigma
        self.use_focal = use_focal
        self.gamma = gamma
        
        self.bin_tool = ProbabilisticBinningTool(num_bins, max_val, device='cpu')
        
        # 权重平滑策略: Log 平滑
        raw_weights = self.bin_tool.class_weights_np
        smoothed_weights = np.log1p(raw_weights) + 1.0 
        self.register_buffer('smooth_weights', torch.tensor(smoothed_weights, dtype=torch.float32))

    def forward(self, logits, target, mask=None):
        """
        Args:
            logits: [B, T, Num_Bins, H, W] 模型输出
            target: [B, T, 1, H, W] 或 [B, T, H, W] 真实物理数值
            mask:   [B, T, 1, H, W] 或 [B, T, H, W] 有效区域掩码
        """
        # 自动设备同步
        if self.bin_tool.device != logits.device:
            self.bin_tool.to(logits.device)
        
        # 1. 构建高斯软标签 (Gaussian Soft Label)
        with torch.no_grad():
            # 确保 target 维度正确 (去除 Channel 维)
            if target.dim() == 5:
                target_squeeze = target.squeeze(2)
            else:
                target_squeeze = target
                
            # 将物理数值转为类别索引
            target_cls = self.bin_tool.to_class(target_squeeze)
            
            # 创建分箱索引网格 [1, 1, 1, 1, Num_Bins]
            bin_indices = torch.arange(self.num_bins, device=logits.device).view(1, 1, 1, 1, -1)
            target_cls_unsqueezed = target_cls.unsqueeze(-1)
            
            # 计算高斯分布
            # dist_sq: 真实类别与所有 bin 的距离平方
            dist_sq = (bin_indices - target_cls_unsqueezed).pow(2)
            soft_target = torch.exp(-dist_sq / (2 * self.sigma ** 2))
            # 归一化为概率分布
            soft_target = soft_target / (soft_target.sum(dim=-1, keepdim=True) + 1e-8)
            
            # 展平以便计算
            soft_target_flat = soft_target.view(-1, self.num_bins)
            target_flat = target_cls.reshape(-1)
            
            # 获取样本权重
            sample_weights = self.smooth_weights.to(logits.device)[target_flat]

        # 2. 计算 Loss
        # logits: [B, T, C, H, W] -> permute -> [B, T, H, W, C] -> reshape -> [N, C]
        B, T, C, H, W = logits.shape
        logits_flat = logits.permute(0, 1, 3, 4, 2).reshape(-1, C)
        log_probs = F.log_softmax(logits_flat, dim=1)
        
        # Soft Cross Entropy: - sum(soft_target * log_prob)
        ce_loss = - (soft_target_flat * log_probs).sum(dim=1)
        
        # Focal Modulation
        if self.use_focal:
            # p_target: 模型在"目标分布"上的置信度
            # 近似为: sum(prob * soft_target)
            probs = torch.exp(log_probs)
            p_target = (probs * soft_target_flat).sum(dim=1)
            focal_weight = (1 - p_target).pow(self.gamma)
            loss = focal_weight * ce_loss
        else:
            loss = ce_loss
            
        # 应用类别平衡权重
        loss = loss * sample_weights
        
        # Mask 处理
        if mask is not None:
            # 确保 mask 维度与 loss (flat) 一致
            if mask.dim() == 5:
                mask = mask.squeeze(2)
            mask_flat = mask.view(-1)
            
            # 只计算 Mask 区域的平均 Loss
            valid_count = mask_flat.sum()
            if valid_count > 0:
                loss = (loss * mask_flat).sum() / valid_count
            else:
                loss = loss.sum() * 0.0 # 避免 NaN
        else:
            loss = loss.mean()
            
        return loss