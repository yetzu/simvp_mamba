import torch
import torch.nn as nn
from .simvp_model import SimVP_Model
from timm.layers.weight_init import trunc_normal_

class ProbabilisticSimVP_Model(SimVP_Model):
    """
    [Refactored] 概率分箱 SimVP-Mamba 模型
    改进：引入 Context-Aware Head (3x3 Conv) 替代简单的 1x1 Readout
    """
    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4, model_type='gSTA',
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True, out_channels=None, 
                 aft_seq_length=None, **kwargs):
        
        # 强制同步 num_bins
        if kwargs.get('num_bins') is not None:
            num_bins = kwargs['num_bins']
            if out_channels != num_bins:
                out_channels = num_bins
            
        super().__init__(in_shape, hid_S, hid_T, N_S, N_T, model_type,
                         mlp_ratio, drop, drop_path, spatio_kernel_enc,
                         spatio_kernel_dec, act_inplace, out_channels, 
                         aft_seq_length, **kwargs)

        # [Refactor] 移除原有的 1x1 readout
        del self.readout
        
        # [New] Context-Aware Head
        # 结构: Conv3x3 (感知邻域) -> Gelu -> Conv1x1 (分类映射)
        self.context_head = nn.Sequential(
            nn.Conv2d(hid_S, hid_S, kernel_size=3, padding=1, groups=1),
            nn.GroupNorm(8, hid_S), # 增加归一化稳定训练
            nn.GELU(),
            nn.Conv2d(hid_S, out_channels, kernel_size=1)
        )

        # 初始化：分类头 bias 置 0
        for m in self.context_head.modules():
            if isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x_raw, **kwargs):
        # 1. 复用 SimVP 的 Encoder 和 Translator
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape 

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B * self.T_out, C_, H_, W_)

        _, C_skip, H_skip, W_skip = skip.shape
        skip = skip.view(B, T, C_skip, H_skip, W_skip)
        skip_last = skip[:, -1:, ...] 
        skip_out = skip_last.expand(-1, self.T_out, -1, -1, -1)
        skip_out = skip_out.reshape(B * self.T_out, C_skip, H_skip, W_skip)
        
        Y = self.dec(hid, skip_out)
        
        # 2. [New] 使用 Context Head 进行输出
        Y = self.context_head(Y)
        
        Y = Y.reshape(B, self.T_out, self.out_channels, H, W)
        return Y