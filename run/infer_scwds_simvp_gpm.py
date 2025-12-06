# run/infer_scwds_simvp_gpm_v4.py
# 基于 V3 修改，引入 Soft-GPM 融合机制以解决双重惩罚问题

import sys
import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

matplotlib.use('Agg')
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metai.utils import MetLabel, MLOGI, MLOGE
from metai.dataset.met_dataloader_scwds import ScwdsDataModule
from metai.model.simvp import SimVPConfig, SimVP
from metai.utils.met_config import get_config

# 竞赛常量
USER_ID = "CP2025000081"
TRACK_ID = "track"
TIME_STEP_MINUTES = 6 

# ==========================================
# GPM 工具类
# ==========================================
class PostProcessor:
    @staticmethod
    def probability_matching(pred_img, ref_source):
        """标准 GPM 计算"""
        pred_flat = pred_img.flatten()
        ref_flat = ref_source.flatten() 
        
        pred_indices = np.argsort(pred_flat)
        ref_sorted = np.sort(ref_flat)
        
        # 线性插值对齐数量
        if len(ref_sorted) != len(pred_flat):
            ref_sorted = np.interp(
                np.linspace(0, 1, len(pred_flat)),
                np.linspace(0, 1, len(ref_sorted)),
                ref_sorted
            )
            
        matched_flat = np.zeros_like(pred_flat)
        matched_flat[pred_indices] = ref_sorted
        
        return matched_flat.reshape(pred_img.shape)

# ==========================================
# 辅助函数
# ==========================================
def find_latest_ckpt(save_dir: str) -> str:
    best = os.path.join(save_dir, 'best.ckpt')
    if os.path.exists(best): return best
    last = os.path.join(save_dir, 'last.ckpt')
    if os.path.exists(last): return last
    cpts = sorted(glob.glob(os.path.join(save_dir, '*.ckpt')))
    if len(cpts) == 0:
        raise FileNotFoundError(f'No checkpoint found in {save_dir}')
    return cpts[-1]

def plot_inference(obs_seq, pred_seq, save_path):
    T_in = obs_seq.shape[0]
    T_out = pred_seq.shape[0]
    cols = max(T_in, T_out)
    fig, axes = plt.subplots(2, cols, figsize=(cols * 1.5, 3.5), constrained_layout=True)
    vmax = obs_seq.max() if obs_seq.max() > 1 else 1.0
    
    for t in range(cols):
        # Input
        ax = axes[0, t]
        if t < T_in:
            ax.imshow(obs_seq[t], cmap='turbo', vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if t == 0: ax.set_title('Obs (Past)', fontsize=10)
        else: ax.axis('off')

        # Pred
        ax = axes[1, t]
        if t < T_out:
            ax.imshow(pred_seq[t], cmap='turbo', vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if t == 0: ax.set_title('Pred (Future)', fontsize=10)
        else: ax.axis('off')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser(description='Infer SCWDS SimVP Model with Soft-GPM (V4)')
    parser.add_argument('--data_path', type=str, default='data/samples.testset.jsonl')
    parser.add_argument('--in_shape', type=int, nargs=4, default=[20, 11, 256, 256])
    parser.add_argument('--save_dir', type=str, default='./output/simvp')
    parser.add_argument('--accelerator', type=str, default='cuda')
    parser.add_argument('--vis', action='store_true', help='Enable visualization')
    parser.add_argument('--vis_output', type=str, default='./output/simvp/vis_infer_gpm_v4')
    
    # [新增] Soft-GPM 核心参数
    parser.add_argument('--gpm_alpha', type=float, default=0.5, 
                        help='GPM 融合初始权重 (0.0=原模型, 1.0=纯GPM, 推荐 0.4-0.6)')
    parser.add_argument('--gpm_decay', type=float, default=0.98, 
                        help='GPM 权重随时间的衰减系数 (推荐 0.95-0.99)')
    return parser.parse_args()

def main():
    args = parse_args()
    met_config = get_config() 
    
    # 1. Config
    config = SimVPConfig(
        data_path=args.data_path,
        in_shape=tuple(args.in_shape),
        save_dir=args.save_dir,
    )
    resize_shape = (config.in_shape[2], config.in_shape[3])
    device = torch.device(args.accelerator if torch.cuda.is_available() else 'cpu')
    FMT = met_config.file_date_format
    
    # 物理最大值 (30.0 mm)
    PHYSICAL_MAX_MM = MetLabel.RA.max / 10.0 
    
    MLOGI(f"[INFO] Start Soft-GPM Inference. Device: {device}")
    MLOGI(f"[INFO] Config: Alpha={args.gpm_alpha}, Decay={args.gpm_decay}")

    # 2. Data
    data_module = ScwdsDataModule(
        data_path=config.data_path,
        resize_shape=resize_shape,
        batch_size=1,
        num_workers=1
    )
    infer_loader = data_module.infer_dataloader()
    
    # 3. Model
    try:
        ckpt_path = find_latest_ckpt(config.save_dir)
        MLOGI(f"加载检查点: {ckpt_path}")
        model = SimVP.load_from_checkpoint(ckpt_path, map_location=device)
        model.eval().to(device)
    except Exception as e:
        MLOGE(f"模型加载失败: {e}")
        return

    # 4. Inference Loop
    with torch.no_grad():
        for bidx, (metadata_list, batch_x, input_mask) in enumerate(infer_loader):
            try:
                batch_x = batch_x.to(device)
                
                # Inference
                batch_y = model.infer_step((metadata_list, batch_x, input_mask), batch_idx=bidx)
                batch_y = batch_y[0, :, 0, :, :] 
                
                metadata = metadata_list[0]
                sample_id = metadata['sample_id']
                case_id = metadata.get('case_id')
                sample_id_parts = sample_id.split('_')
                task_id = metadata.get('task_id')
                region_id = metadata['region_id']
                station_id = metadata['station_id']
                time_id = sample_id_parts[2] 
                timestamps = metadata['timestamps']
                
                last_obs_idx = batch_x.shape[1] - 1
                if last_obs_idx >= len(timestamps):
                    last_obs_idx = -1
                
                last_obs_dt = datetime.strptime(timestamps[last_obs_idx], FMT)

                # =========================================================
                # [Soft-GPM 策略] 时序池化 + 强度保护 + 柔性融合
                # =========================================================
                
                # A. 提取参考源 (归一化值 0~1)
                input_seq_tensor = batch_x[0, :, 0, :, :] 
                ref_source = input_seq_tensor.cpu().numpy()
                
                # B. 强度保护
                ref_max_norm = ref_source.max()
                # 设定 1.0mm 物理阈值作为保护线
                THRESHOLD_MM_PHYSICAL = 1.0 
                DISABLE_GPM_THRESHOLD = THRESHOLD_MM_PHYSICAL / PHYSICAL_MAX_MM
                
                base_apply_gpm = True
                if ref_max_norm < DISABLE_GPM_THRESHOLD:
                    base_apply_gpm = False

                pred_frames_vis = []
                seq_mean_val_stored = 0.0
                
                for idx, y in enumerate(batch_y):
                    # 1. 插值原始预测
                    y_interp = F.interpolate(
                        y.unsqueeze(0).unsqueeze(0), 
                        size=(301, 301),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                    y_raw_np = y_interp.cpu().numpy() # 原始预测 (Raw)

                    # 2. 计算 GPM 预测
                    y_final_np = y_raw_np
                    
                    if base_apply_gpm:
                        # 生成全量 GPM 结果
                        y_gpm_np = PostProcessor.probability_matching(y_raw_np, ref_source)
                        
                        # [关键] 计算当前帧的融合权重 alpha
                        # idx=0 (6min): alpha * decay^0 = alpha
                        # idx=19 (120min): alpha * decay^19 (变小)
                        current_alpha = args.gpm_alpha * (args.gpm_decay ** idx)
                        
                        # [关键] 线性融合
                        # 既保留 Raw 的位置底座，又利用 GPM 拉升峰值
                        y_final_np = (1.0 - current_alpha) * y_raw_np + current_alpha * y_gpm_np

                    # 3. 阈值清理 (清理极小底噪)
                    y_final_np[y_final_np < 0.003] = 0.0
                    
                    # 4. 还原为存储数值 (0-300)
                    y_save = (y_final_np * MetLabel.RA.max).astype(np.float32)
                    
                    # 保存
                    forecast_dt = last_obs_dt + timedelta(minutes=TIME_STEP_MINUTES * (idx + 1))
                    forecast_time_str = forecast_dt.strftime(FMT)
                    
                    npy_dir = os.path.join('submit', 'output', USER_ID, TRACK_ID, case_id)
                    os.makedirs(npy_dir, exist_ok=True)
                    
                    npy_path = os.path.join(
                        npy_dir,
                        f"{task_id}_{region_id}_{time_id}_{station_id}_Forcast_{forecast_time_str}.npy"
                    )
                    
                    np.save(npy_path, y_save)
                    
                    seq_mean_val_stored += float(y_save.mean())
                    
                    if args.vis:
                        pred_frames_vis.append(y_save / 300)
                
                seq_mean_val_stored /= len(batch_y)
                
                gpm_status = f"Soft(A={args.gpm_alpha})" if base_apply_gpm else "OFF(Safety)"
                MLOGI(f"No.{bidx} {sample_id} | {gpm_status} | Mean: {seq_mean_val_stored/10.0:.4f}mm")

                if args.vis:
                    obs_frames = batch_x[0, :, 0, :, :].cpu().numpy()
                    pred_frames = np.array(pred_frames_vis)
                    vis_path = os.path.join(args.vis_output, f"{sample_id}.png")
                    plot_inference(obs_frames, pred_frames, vis_path)

            except Exception as e:
                import traceback
                traceback.print_exc()
                MLOGE(f"样本 {bidx} 推理失败: {e}")
                continue
            
    MLOGI("✅ Soft-GPM (V4) 推理全部完成！")

if __name__ == '__main__':
    main()