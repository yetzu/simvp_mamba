# metai/model/simvp/prob_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ProbabilisticBinningTool:
    """
    概率分箱工具类：
    负责生成对齐比赛阈值的非均匀 Bin，进行数值/类别互转，并计算加权 Loss 所需的类别权重。
    
    设计原则：
    1. Bin Edges 精确对齐比赛的关键阈值: [0.1, 1.0, 2.0, 5.0, 8.0] mm。
    2. 类别权重 W_cls 针对 >=8.0mm 的强降水区域进行了 20 倍的定向增强。
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
        
        # 计算 Centers (用于 Argmax 解码)
        self.centers_np = (self.edges_np[:-1] + self.edges_np[1:]) / 2.0
        
        # --- 2. 类别权重设计：匹配比赛规则，定向增强强降水 ---
        self.class_weights_np = self._calculate_weights()
        
        # 转换为 Tensor，并放置到指定设备
        self.edges = torch.tensor(self.edges_np, dtype=torch.float32).to(device)
        self.centers = torch.tensor(self.centers_np, dtype=torch.float32).to(device)
        self.class_weights = torch.tensor(self.class_weights_np, dtype=torch.float32).to(device)

    # [Fix] 新增 to 方法以修复 AttributeError
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
        
        # 查找关键阈值对应的 Bin 索引 (使用 right side 确保阈值本身被视为高一级)
        idx_01 = np.searchsorted(self.edges_np, 0.1, side='right') - 1
        idx_1  = np.searchsorted(self.edges_np, 1.0, side='right') - 1
        idx_2  = np.searchsorted(self.edges_np, 2.0, side='right') - 1
        idx_5  = np.searchsorted(self.edges_np, 5.0, side='right') - 1
        idx_8  = np.searchsorted(self.edges_np, 8.0, side='right') - 1
        
        # 赋予权重 (根据图 2 表 2，并进行增强)
        weights[idx_01:idx_1] = 2.0  # 0.1~1.0mm (权重 0.1 -> 增强 2x)
        weights[idx_1:idx_2]  = 3.0  # 1.0~2.0mm (权重 0.1 -> 增强 3x)
        weights[idx_2:idx_5]  = 5.0  # 2.0~5.0mm (权重 0.2 -> 增强 5x)
        weights[idx_5:idx_8]  = 10.0 # 5.0~8.0mm (权重 0.25 -> 增强 10x)
        weights[idx_8:]       = 20.0 # >=8.0mm (权重 0.35 -> 增强 20x)
        
        return weights

    def to_class(self, value_tensor):
        """将连续降水值 [0, 30] 映射为类别索引 [0, num_classes-1]"""
        # 确保 edges 在正确的设备上
        edges_on_device = self.edges.to(value_tensor.device)
        idxs = torch.bucketize(value_tensor, edges_on_device) - 1
        return torch.clamp(idxs, 0, self.num_classes - 1)

    def class_to_value(self, class_idxs):
        """将类别索引还原为降水值 (Argmax 解码)"""
        return self.centers.to(class_idxs.device)[class_idxs]

class ProbabilisticCrossEntropyLoss(nn.Module):
    """
    概率分箱模式下的加权交叉熵 Loss
    使用 ProbabilisticBinningTool 提供的类别权重进行训练
    """
    def __init__(self, num_bins=64, max_val=30.0):
        super().__init__()
        self.num_bins = num_bins
        self.max_val = max_val
        # 延迟初始化 bin_tool，确保在 device 切换后能正确创建
        self.bin_tool = ProbabilisticBinningTool(num_bins, max_val, device='cpu')

    def forward(self, logits, target, mask=None):
        # 动态创建/移动 bin_tool 到当前设备 (保证鲁棒性)
        if self.bin_tool.device != logits.device:
            # 优先尝试使用 to 方法（如果已添加），否则重新实例化
            if hasattr(self.bin_tool, 'to'):
                self.bin_tool.to(logits.device)
            else:
                self.bin_tool = ProbabilisticBinningTool(
                    self.num_bins, 
                    self.max_val, 
                    device=logits.device
                )
        
        # 1. Target 转换为类别 (不可导)
        with torch.no_grad():
            target_squeeze = target.squeeze(2) # [B, T, H, W]
            target_cls = self.bin_tool.to_class(target_squeeze)
            
        B, T, C, H, W = logits.shape
        
        # 2. Reshape Logits 和 Target 为 [N, C] 和 [N]
        logits_flat = logits.permute(0, 1, 3, 4, 2).reshape(-1, C)
        target_flat = target_cls.reshape(-1)
        
        # 3. 计算加权 CE Loss
        class_weights = self.bin_tool.class_weights.to(logits.device)
        loss = F.cross_entropy(
            logits_flat, 
            target_flat, 
            weight=class_weights,
            reduction='none'
        )
        
        # 4. 应用 Mask
        if mask is not None:
            mask_flat = mask.view(-1)
            loss = (loss * mask_flat).sum() / (mask_flat.sum() + 1e-6)
        else:
            loss = loss.mean()
            
        return loss