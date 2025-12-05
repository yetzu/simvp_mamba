# metai/model/simvp/prob_model.py

from .simvp_model import SimVP_Model
from typing import Dict, Any

class ProbabilisticSimVP_Model(SimVP_Model):
    """
    概率分箱 SimVP-Mamba 模型。
    继承自 SimVP_Model，其核心结构（Encoder-Mamba-Decoder）不变。
    模型的输出通道数 (out_channels) 被设置为概率分箱的数量 (num_bins)。
    输出为 Logits: [B, T_out, Num_Bins, H, W]
    """
    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4, model_type='gSTA',
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True, out_channels=None, 
                 aft_seq_length=None, **kwargs):
        
        # 确保 out_channels 使用了 num_bins 的值 (在配置中已处理)
        # 如果配置中传入了 num_bins，则优先使用它
        if kwargs.get('num_bins') is not None and out_channels != kwargs['num_bins']:
            out_channels = kwargs['num_bins']
            
        super().__init__(in_shape, hid_S, hid_T, N_S, N_T, model_type,
                         mlp_ratio, drop, drop_path, spatio_kernel_enc,
                         spatio_kernel_dec, act_inplace, out_channels, 
                         aft_seq_length, **kwargs)

    def forward(self, x_raw, **kwargs):
        """
        前向传播保持不变，输出 Logits 矩阵 [B, T_out, Num_Bins, H, W]。
        """
        return super().forward(x_raw, **kwargs)