# infer_scwds_prob.py (基于 run/infer_scwds_simvp.py 修改，适配概率分箱模式)

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

# 设置 matplotlib 后端
matplotlib.use('Agg')

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metai.utils import MetLabel, MLOGI, MLOGE
from metai.dataset.met_dataloader_scwds import ScwdsDataModule
from metai.model.simvp.simvp_config import SimVPConfig
# [修改点 1] 导入 Probabilistic SimVP Trainer
from metai.model.simvp.prob_trainer import ProbabilisticSimVP 
from metai.utils.met_config import get_config

# 竞赛常量
USER_ID = "CP2025000081"
TRACK_ID = "track1"
TIME_STEP_MINUTES = 6 

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
    """绘制推理结果对比图 (输入输出均为 numpy 数组)"""
    T_in = obs_seq.shape[0]
    T_out = pred_seq.shape[0]
    cols = max(T_in, T_out)
    fig, axes = plt.subplots(2, cols, figsize=(cols * 1.5, 3.5), constrained_layout=True)
    
    # 统一归一化显示 (假设输入已经是 0-1 或 0-255，这里统一按 0-1 显示)
    vmax = obs_seq.max() if obs_seq.max() > 1 else 1.0
    
    for t in range(cols):
        # Input
        ax = axes[0, t]
        if t < T_in:
            ax.imshow(obs_seq[t], cmap='turbo', vmin=0, vmax=vmax)
            if t == 0: ax.set_title('Obs (Past)', fontsize=10)
        else: ax.axis('off')
        ax.axis('off')

        # Pred
        ax = axes[1, t]
        if t < T_out:
            ax.imshow(pred_seq[t], cmap='turbo', vmin=0, vmax=vmax)
            if t == 0: ax.set_title('Pred (Future)', fontsize=10)
        else: ax.axis('off')
        ax.axis('off')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser(description='Infer SCWDS Probabilistic SimVP Model')
    parser.add_argument('--data_path', type=str, default='data/samples.testset.jsonl')
    parser.add_argument('--in_shape', type=int, nargs=4, default=[20, 11, 256, 256])
    parser.add_argument('--save_dir', type=str, default='./output/prob_simvp')
    parser.add_argument('--accelerator', type=str, default='cuda')
    parser.add_argument('--vis', action='store_true', help='Enable visualization')
    parser.add_argument('--vis_output', type=str, default='./output/prob_simvp/vis_infer')
    
    # [新增] 分箱参数
    parser.add_argument('--num_bins', type=int, default=64, help='概率分箱的数量')
    return parser.parse_args()

def main():
    args = parse_args()
    met_config = get_config() 
    
    # 1. Config
    config = SimVPConfig(
        data_path=args.data_path,
        in_shape=tuple(args.in_shape),
        save_dir=args.save_dir,
        out_channels=args.num_bins, # 设置 out_channels
        num_bins=args.num_bins       # 注入 num_bins
    )
    resize_shape = (config.in_shape[2], config.in_shape[3])
    device = torch.device(args.accelerator if torch.cuda.is_available() else 'cpu')
    FMT = met_config.file_date_format
    
    MLOGI(f"[INFO] Start Probabilistic Inference. Device: {device}, SaveDir: {args.save_dir}")

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
        MLOGI(f"加载检查点: {ckpt_path} (Num Bins: {config.num_bins})")
        # [修改点 2] 使用 ProbabilisticSimVP 加载模型
        model = ProbabilisticSimVP.load_from_checkpoint(ckpt_path, map_location=device, num_bins=config.num_bins)
        model.eval().to(device)
    except Exception as e:
        MLOGE(f"模型加载失败: {e}")
        return

    # 4. Inference Loop
    with torch.no_grad():
        for bidx, (metadata_list, batch_x, input_mask) in enumerate(infer_loader):
            try:
                batch_x = batch_x.to(device)
                
                # Inference: batch_y 已是 Argmax 解码后的归一化数值 [B, T, 1, H, W]
                # ProbabilisticSimVP.infer_step 实现了 Argmax 解码和反归一化到 [0, 1]
                batch_y = model.infer_step((metadata_list, batch_x, input_mask), batch_idx=bidx)
                
                # 处理维度: 去掉 Batch=1 和 Channel=1 -> [20, 256, 256]
                batch_y = batch_y[0, :, 0, :, :] 
                
                metadata = metadata_list[0]
                sample_id = metadata['sample_id']
                
                # 解析元数据
                sample_id_parts = sample_id.split('_')
                task_id = metadata.get('task_id')
                region_id = metadata['region_id']
                station_id = metadata['station_id']
                time_id = sample_id_parts[2] 
                case_id = metadata.get('case_id')
                timestamps = metadata['timestamps']

                if not timestamps: continue
                
                # 计算起始预测时间
                last_obs_idx = batch_x.shape[1] - 1
                if last_obs_idx >= len(timestamps):
                    last_obs_idx = -1
                
                last_obs_dt = datetime.strptime(timestamps[last_obs_idx], FMT)
                
                pred_frames_vis = []
                
                # 用于统计该样本的整体预测情况
                seq_max_val = 0.0
                seq_mean_val = 0.0
                seq_zero_count = 0 
                seq_total_count = 0
                
                for idx, y in enumerate(batch_y):
                    # 1. 插值到 301x301
                    y_interp = F.interpolate(
                        y.unsqueeze(0).unsqueeze(0), 
                        size=(301, 301),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze() # -> (301, 301)
                    
                    # 2. Argmax 解码后的 y 已经是 [0, 1] 归一化，
                    # 需乘以 MetLabel.RA.max (300) 还原为 mm
                    y_np = y_interp.cpu().numpy()

                    # 还原为 mm 级降水并进行后处理
                    y_save = (y_np * MetLabel.RA.max).astype(np.float32)
                    
                    # 构造保存路径
                    forecast_dt = last_obs_dt + timedelta(minutes=TIME_STEP_MINUTES * (idx + 1))
                    forecast_time_str = forecast_dt.strftime(FMT)
                    
                    npy_dir = os.path.join(
                        'submit', 'output', USER_ID, TRACK_ID, case_id
                    )
                    os.makedirs(npy_dir, exist_ok=True)
                    
                    npy_path = os.path.join(
                        npy_dir,
                        f"{task_id}_{region_id}_{time_id}_{station_id}_Forcast_{forecast_time_str}.npy"
                    )
                    
                    np.save(npy_path, y_save)
                    
                    # 统计 (假设 MetLabel.RA.max 是 300.0)
                    seq_max_val = max(seq_max_val, float(y_save.max()) / 10.0)
                    seq_mean_val += float(y_save.mean()) / 10.0
                    seq_zero_count += np.sum(y_save == 0)
                    seq_total_count += y_save.size
                    
                    # 收集可视化数据 (此时还是 0-1 的 float)
                    if args.vis:
                        pred_frames_vis.append(y_save / 300)
                
                # 打印关键统计信息
                seq_mean_val /= len(batch_y)
                zero_ratio = seq_zero_count / seq_total_count if seq_total_count > 0 else 0.0
                MLOGI(f"No.{bidx} {sample_id} | Max: {seq_max_val:.2f}mm | Mean: {seq_mean_val:.4f}mm | Zero: {zero_ratio:.2%}")

                # 可视化 (如果在参数中启用)
                if args.vis:
                    # obs_frames 仍需手动归一化
                    obs_frames = batch_x[0, :, 0, :, :].cpu().numpy()
                    pred_frames = np.array(pred_frames_vis)
                    vis_path = os.path.join(args.vis_output, f"{sample_id}.png")
                    plot_inference(obs_frames, pred_frames, vis_path)

            except Exception as e:
                import traceback
                traceback.print_exc()
                MLOGE(f"样本 {bidx} 推理失败: {e}")
                continue
            
    MLOGI("✅ 推理全部完成！")

if __name__ == '__main__':
    main()