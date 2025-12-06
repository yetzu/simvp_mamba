# metai/model/simvp/prob_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ProbabilisticBinningTool:
    """
    概率分箱工具类：
    负责生成对齐比赛阈值的非均匀 Bin，进行数值/类别互转，并计算加权 Loss 所需的类别权重。
    """
    def __init__(self, num_bins=64, max_val=30.0, device='cpu'):
        # 属性保存
        self.num_bins = num_bins
        self.max_val = max_val
        self.device = device
        
        # --- 1. 精心设计对齐比赛阈值的 Bin Edges ---
        # 区间 1: 0.0 ~ 1.0mm (10 个区间，分辨率 0.1mm)
        edges_1 = np.linspace(0.0, 1.0, 11) 
        # 区间 2: 1.0 ~ 8.0mm (20 个区间，分辨率 0.35mm)
        edges_2 = np.linspace(1.0, 8.0, 21)
        # 区间 3: 8.0 ~ 30.0mm (约 34 个区间，重点覆盖高值区)
        edges_3 = np.linspace(8.0, max_val, num_bins - 30 + 1)
        
        all_edges = np.concatenate([edges_1, edges_2, edges_3])
        self.edges_np = np.unique(all_edges)
        
        # 最终 Bin 数：确保数量不大于期望值
        if len(self.edges_np) > num_bins + 1:
             self.edges_np = self.edges_np[:num_bins+1]
        
        self.num_classes = len(self.edges_np) - 1
        
        # 计算 Centers (用于 Argmax/Expectation 解码)
        self.centers_np = (self.edges_np[:-1] + self.edges_np[1:]) / 2.0
        self.centers_np[0] = 0.0
        
        # --- 2. 类别权重设计 ---
        self.class_weights_np = self._calculate_weights()
        
        # 转换为 Tensor，并放置到指定设备
        self.edges = torch.tensor(self.edges_np, dtype=torch.float32).to(device)
        self.centers = torch.tensor(self.centers_np, dtype=torch.float32).to(device)
        self.class_weights = torch.tensor(self.class_weights_np, dtype=torch.float32).to(device)

    def to(self, device):
        """将工具类内部的 Tensor 移动到指定设备"""
        self.device = device
        self.edges = self.edges.to(device)
        self.centers = self.centers.to(device)
        self.class_weights = self.class_weights.to(device)
        return self

    def _calculate_weights(self):
        """计算并返回权重 numpy 数组"""
        weights = np.ones(self.num_classes, dtype=np.float32)
        
        # 查找关键阈值对应的 Bin 索引
        idx_01 = np.searchsorted(self.edges_np, 0.1, side='right') - 1
        idx_1  = np.searchsorted(self.edges_np, 1.0, side='right') - 1
        idx_2  = np.searchsorted(self.edges_np, 2.0, side='right') - 1
        idx_5  = np.searchsorted(self.edges_np, 5.0, side='right') - 1
        idx_8  = np.searchsorted(self.edges_np, 8.0, side='right') - 1
        
        # [Fix] 优化权重策略：
        # 1. 显著降低 Class 0 (0.0-0.1mm) 的权重，防止模型为了降低总 Loss 而倾向于预测全 0。
        # 2. 保持对强降水的极高关注。
        
        weights[0:idx_01] = 0.05   # 无雨/微量雨：降权至 0.05 (非常重要！)
        weights[idx_01:idx_1] = 2.0  # 0.1~1.0mm
        weights[idx_1:idx_2]  = 3.0  # 1.0~2.0mm
        weights[idx_2:idx_5]  = 5.0  # 2.0~5.0mm
        weights[idx_5:idx_8]  = 10.0 # 5.0~8.0mm
        weights[idx_8:]       = 20.0 # >=8.0mm (强降水)
        
        return weights

    def to_class(self, value_tensor):
        """将连续降水值 [0, 30] 映射为类别索引 [0, num_classes-1]"""
        edges_on_device = self.edges.to(value_tensor.device)
        idxs = torch.bucketize(value_tensor, edges_on_device) - 1
        return torch.clamp(idxs, 0, self.num_classes - 1)

    def class_to_value(self, class_idxs):
        """将类别索引还原为降水值"""
        return self.centers.to(class_idxs.device)[class_idxs]

class ProbabilisticCrossEntropyLoss(nn.Module):
    """
    [改进版] 概率分箱损失函数 v2
    集成 Gaussian Soft Labels (解决序数问题) + Focal Loss (解决虚警与权重过激问题)
    """
    def __init__(self, num_bins=64, max_val=30.0, sigma=2.0, use_focal=True, gamma=2.0):
        super().__init__()
        self.num_bins = num_bins
        self.max_val = max_val
        self.sigma = sigma          # 高斯分布标准差：控制软标签的"宽窄"，建议 1.0~2.0
        self.use_focal = use_focal  # 是否启用 Focal Loss：强烈建议启用
        self.gamma = gamma          # Focal聚焦参数：通常取 2.0
        
        # 初始化工具类
        self.bin_tool = ProbabilisticBinningTool(num_bins, max_val, device='cpu')
        
        # [优化] 重算更温和的权重，避免 400x 差异导致梯度爆炸
        # 原逻辑: 0.05 -> 20.0 (400倍)
        # 新逻辑: 对数平滑 + 适度增强高值区 (约 1.0 -> 5.0)
        raw_weights = self.bin_tool.class_weights_np
        smoothed_weights = np.log1p(raw_weights) + 0.5 
        smoothed_weights[-10:] *= 2.0  # 依然对强降水保持关注，但不极端
        
        self.register_buffer('smooth_weights', torch.tensor(smoothed_weights, dtype=torch.float32))

    def forward(self, logits, target, mask=None):
        """
        logits: [B, T, Num_Bins, H, W]
        target: [B, T, 1, H, W] (连续物理值)
        """
        # 自动设备同步
        if self.bin_tool.device != logits.device:
            if hasattr(self.bin_tool, 'to'): self.bin_tool.to(logits.device)
            else: self.bin_tool = ProbabilisticBinningTool(self.num_bins, self.max_val, device=logits.device)
        
        # 1. 生成高斯软标签 (Gaussian Soft Target)
        with torch.no_grad():
            target_squeeze = target.squeeze(2) # [B, T, H, W]
            target_cls = self.bin_tool.to_class(target_squeeze) # [B, T, H, W] (Index)
            
            # 创建 bin 索引网格，计算每个 Bin 与 真值 Bin 的距离
            # [1, 1, 1, 1, Num_Bins]
            bin_indices = torch.arange(self.num_bins, device=logits.device).view(1, 1, 1, 1, -1)
            target_cls_unsqueezed = target_cls.unsqueeze(-1)
            
            # 高斯分布核心: exp(- dist^2 / 2sigma^2)
            dist_sq = (bin_indices - target_cls_unsqueezed).pow(2)
            soft_target = torch.exp(-dist_sq / (2 * self.sigma ** 2))
            
            # 归一化 (Sum=1)
            soft_target = soft_target / (soft_target.sum(dim=-1, keepdim=True) + 1e-8)
            
            # 展平 [N, Num_Bins]
            soft_target_flat = soft_target.view(-1, self.num_bins)
            
            # 获取对应的平滑类别权重 [N]
            target_flat = target_cls.reshape(-1)
            sample_weights = self.smooth_weights.to(logits.device)[target_flat]

        # 2. 计算预测概率
        B, T, C, H, W = logits.shape
        # [N, C]
        logits_flat = logits.permute(0, 1, 3, 4, 2).reshape(-1, C)
        log_probs = F.log_softmax(logits_flat, dim=1) # Log Softmax
        
        # 3. 计算基础交叉熵 (Soft Target Cross Entropy)
        # CE = - sum(target * log_prob)
        ce_per_class = - (soft_target_flat * log_probs)
        
        # 4. 应用 Focal Loss 机制
        if self.use_focal:
            # weight = (1 - p)^gamma
            # p 是模型对该类别的预测概率。这里近似使用 exp(log_probs)
            probs = torch.exp(log_probs)
            # 这里的 Focal 权重是针对每个类别的，不仅针对正确类别
            focal_weight = (1 - probs).pow(self.gamma)
            
            # 最终 Loss = Focal_Weight * Soft_Target * CE
            loss_per_sample = (focal_weight * ce_per_class).sum(dim=1)
        else:
            loss_per_sample = ce_per_class.sum(dim=1)
            
        # 5. 应用类别平衡权重 & Mask
        loss_per_sample = loss_per_sample * sample_weights
        
        if mask is not None:
            mask_flat = mask.view(-1)
            loss = (loss_per_sample * mask_flat).sum() / (mask_flat.sum() + 1e-6)
        else:
            loss = loss_per_sample.mean()
            
        return loss