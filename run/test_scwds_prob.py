# run/test_scwds_prob.py (基于 run/test_scwds_simvp.py 修改，适配概率分箱模式 + 基座对比)

import sys
import os
import glob
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import numpy as np
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
matplotlib.use('Agg')

from metai.dataset.met_dataloader_scwds import ScwdsDataModule
from metai.model.simvp.simvp_config import SimVPConfig
# [修改点 1] 导入 Probabilistic SimVP Trainer
from metai.model.simvp.prob_trainer import ProbabilisticSimVP 
from metai.model.simvp.simvp_trainer import SimVP as SimVP_Regression # 导入原 SimVP 以复用 MetricConfig 和加载基座

# ==========================================
# Part 0: 辅助工具函数
# ==========================================

class TeeLogger:
    """同时输出到控制台和文件的日志类"""
    def __init__(self, log_file_path):
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        self.console = sys.stdout
        
    def write(self, message):
        self.console.write(message)
        self.log_file.write(message)
        self.log_file.flush()
        
    def flush(self):
        self.console.flush()
        self.log_file.flush()
        
    def close(self):
        if self.log_file:
            self.log_file.close()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# 全局日志对象，初始为 None
_logger = None

def set_logger(log_file_path):
    global _logger
    _logger = TeeLogger(log_file_path)
    sys.stdout = _logger
    
def restore_stdout():
    global _logger
    if _logger:
        sys.stdout = _logger.console
        _logger.close()
        _logger = None

def find_best_ckpt(save_dir: str) -> str:
    # 优先查找 best.ckpt
    best = os.path.join(save_dir, 'best.ckpt')
    if os.path.exists(best): return best
    
    # 其次查找 last.ckpt
    last = os.path.join(save_dir, 'last.ckpt')
    if os.path.exists(last): return last
    
    # 最后查找所有 checkpoint 文件，返回最新的
    cpts = sorted(glob.glob(os.path.join(save_dir, '*.ckpt')))
    if len(cpts) == 0:
        raise FileNotFoundError(f'No checkpoint found in {save_dir}')
    return cpts[-1]

def get_checkpoint_info(ckpt_path: str):
    """从 checkpoint 文件中提取训练关键信息"""
    try:
        # 使用 SimVP_Regression 的 load_from_checkpoint 逻辑来加载信息
        ckpt = torch.load(ckpt_path, map_location='cpu')
        epoch = ckpt.get('epoch', None)
        global_step = ckpt.get('global_step', None)
        hparams = ckpt.get('hyper_parameters', {})
        return {
            'epoch': epoch,
            'global_step': global_step,
            'hparams': hparams,
            'ckpt_name': os.path.basename(ckpt_path)
        }
    except Exception as e:
        return {'error': str(e)}

def print_checkpoint_info(ckpt_info: dict):
    """打印 checkpoint 信息"""
    if 'error' in ckpt_info:
        print(f"[WARNING] 无法读取 checkpoint 信息: {ckpt_info['error']}")
        return
    
    print("=" * 80)
    print(f"[INFO] Loaded Checkpoint: {ckpt_info['ckpt_name']}")
    print(f"  Epoch: {ckpt_info.get('epoch', 'N/A')}")
    print(f"  Global Step: {ckpt_info.get('global_step', 'N/A')}")
    
    hparams = ckpt_info.get('hparams', {})
    if hparams:
        print(f"  Model Type: {hparams.get('model_type', 'N/A')}")
        print(f"  In Shape: {hparams.get('in_shape', 'N/A')}")
        print(f"  Out Seq Length: {hparams.get('aft_seq_length', 'N/A')}")
        print(f"  Prob Bins: {hparams.get('num_bins', 'N/A')}") # [修改点 0] 打印 num_bins
    print("=" * 80)

# ==========================================
# Part 1: 全局评分配置 (Metric Configuration) - 复用 SimVP Trainer 的常量
# ==========================================
# 注意：我们复用原 SimVP Trainer 的 MetricConfig，因为它与比赛规则一致
class MetricConfig:
    MM_MAX = 30.0
    THRESHOLD_NOISE = 0.05 
    LEVEL_EDGES = np.array([0.1, 1.0, 2.0, 5.0, 8.0, np.inf], dtype=np.float32)
    _raw_level_weights = np.array([0.1, 0.1, 0.2, 0.25,  0.35], dtype=np.float32)
    LEVEL_WEIGHTS = _raw_level_weights / _raw_level_weights.sum()

    TIME_WEIGHTS_DICT = {
        0: 0.0075, 1: 0.02, 2: 0.03, 3: 0.04, 4: 0.05,
        5: 0.06, 6: 0.07, 7: 0.08, 8: 0.09, 9: 0.1,
        10: 0.09, 11: 0.08, 12: 0.07, 13: 0.06, 14: 0.05,
        15: 0.04, 16: 0.03, 17: 0.02, 18: 0.0075, 19: 0.005
    }

    @staticmethod
    def get_time_weights(T):
        if T == 20:
            return np.array([MetricConfig.TIME_WEIGHTS_DICT[t] for t in range(T)], dtype=np.float32)
        else:
            print(f"[WARN] T={T}, expected 20. Using uniform time weights.")
            return np.ones(T, dtype=np.float32) / T

# ==========================================
# Part 2: 核心统计计算 (Core Metrics)
# ==========================================
def calc_seq_metrics(true_seq, pred_seq, verbose=True):
    T, H, W = true_seq.shape
    
    pred_clean = pred_seq.copy()
    pred_clean[pred_clean < (MetricConfig.THRESHOLD_NOISE / MetricConfig.MM_MAX)] = 0.0
    
    time_weights = MetricConfig.get_time_weights(T)
    score_k_list = []
    
    tru_mm_seq = np.clip(true_seq, 0.0, None) * MetricConfig.MM_MAX
    prd_mm_seq = np.clip(pred_clean, 0.0, None) * MetricConfig.MM_MAX
    
    ts_mean_levels = np.zeros(len(MetricConfig.LEVEL_WEIGHTS))
    mae_mm_mean_levels = np.zeros(len(MetricConfig.LEVEL_WEIGHTS))
    corr_sum = 0.0

    if verbose:
        print(f"True Stats (mm): Max={np.max(tru_mm_seq):.2f}, Mean={np.mean(tru_mm_seq):.2f}")
        print(f"Pred Stats (mm): Max={np.max(prd_mm_seq):.2f}, Mean={np.mean(prd_mm_seq):.2f}")
        print("-" * 90)
        print(f"{'T':<3} | {'Corr(R)':<9} | {'TS_w_sum':<9} | {'Score_k':<9} | {'W_time':<8}")
        print("-" * 90)

    for t in range(T):
        tru_frame = tru_mm_seq[t].reshape(-1)
        prd_frame = prd_mm_seq[t].reshape(-1)
        abs_err = np.abs(prd_frame - tru_frame)

        mask_valid_corr = (tru_frame > 0) | (prd_frame > 0)
        if mask_valid_corr.sum() > 1:
            t_valid = tru_frame[mask_valid_corr]
            p_valid = prd_frame[mask_valid_corr]
            numerator = np.sum((t_valid - t_valid.mean()) * (p_valid - p_valid.mean()))
            denom = np.sqrt(np.sum((t_valid - t_valid.mean())**2) * np.sum((p_valid - p_valid.mean())**2))
            R_k = numerator / (denom + 1e-8)
        else:
            R_k = 0.0
        
        R_k = float(np.clip(R_k, -1.0, 1.0))
        corr_sum += R_k
        term_corr = np.sqrt(np.exp(R_k - 1.0))

        weighted_sum_metrics = 0.0
        for i in range(len(MetricConfig.LEVEL_WEIGHTS)):
            low = MetricConfig.LEVEL_EDGES[i]
            high = MetricConfig.LEVEL_EDGES[i+1]
            w_i = MetricConfig.LEVEL_WEIGHTS[i]

            tru_bin = (tru_frame >= low) & (tru_frame < high)
            prd_bin = (prd_frame >= low) & (prd_frame < high)
            
            tp = np.logical_and(tru_bin, prd_bin).sum()
            fn = np.logical_and(tru_bin, ~prd_bin).sum()
            fp = np.logical_and(~tru_bin, prd_bin).sum()
            denom_ts = tp + fn + fp
            ts_val = (tp / denom_ts) if denom_ts > 0 else 1e-6
            
            mask_eval = tru_bin | prd_bin
            mae_val = np.mean(abs_err[mask_eval]) if mask_eval.sum() > 0 else 0.0
            
            ts_mean_levels[i] += ts_val / T
            mae_mm_mean_levels[i] += mae_val / T
            
            term_mae = np.sqrt(np.exp(-mae_val / 100.0))
            weighted_sum_metrics += w_i * ts_val * term_mae

        Score_k = term_corr * weighted_sum_metrics
        score_k_list.append(Score_k)

        if verbose:
            print(f"{t:<3} | {R_k:<9.4f} | {weighted_sum_metrics:<9.4f} | {Score_k:<9.4f} | {time_weights[t]:<8.4f}")

    score_k_arr = np.array(score_k_list)
    final_score = np.sum(score_k_arr * time_weights)
    
    print("-" * 90)
    ts_str = ", ".join([f"{v:.3f}" for v in ts_mean_levels])
    mae_str = ", ".join([f"{v:.3f}" for v in mae_mm_mean_levels])
    
    print(f"[METRIC] TS_mean  (Levels): {ts_str}")
    print(f"[METRIC] MAE_mean (Levels): {mae_str}")
    print(f"[METRIC] Corr_mean: {corr_sum / T:.4f}")
    print(f"[METRIC] Final_Weighted_Score: {final_score:.6f}")
    print(f"[METRIC] Score_per_t: {', '.join([f'{s:.3f}' for s in score_k_arr])}")
    print("-" * 90)
    
    return {
        "final_score": final_score,
        "score_per_frame": score_k_arr,
        "pred_clean": pred_clean
    }

# ==========================================
# Part 3: 绘图功能 (Visualization)
# ==========================================

def create_precipitation_cmap():
    """
    创建自定义降水色标
    区间: 0.01 <= r < 0.1, 0.1 <= r < 1, ..., r >= 8
    0 值 (及 < 0.01) 显示为白色
    """
    hex_colors = [
        '#9CF48D',  # 0.01 <= r < 0.1 (浅绿)
        '#3CB73A',  # 0.1 <= r < 1 (中绿)
        '#63B7FF',  # 1 <= r < 2 (浅蓝)
        '#0200F9',  # 2 <= r < 5 (深蓝)
        '#EE00F0',  # 5 <= r < 8 (紫红)
        '#9F0000'   # r >= 8 (深红)
    ]
    
    cmap = mcolors.ListedColormap(hex_colors)
    cmap.set_bad('white')
    cmap.set_under('white')
    
    # 边界设置：起始值设为0.01，确保0值落入under区域（白色）
    bounds = [0.01, 0.1, 1, 2, 5, 8, 100]
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=len(hex_colors))
    
    return cmap, norm

def plot_seq_visualization(obs_seq, true_seq, pred_seq, base_seq, scores, out_path, vmax=1.0):
    """
    绘制 Obs, GT, Base Pred, Prob Pred, Diff 对比图
    [修改] 增加 base_seq 参数，如果非 None，则绘制 5 行，否则绘制 4 行
    """
    T = true_seq.shape[0] # T_out = 20
    rows = 5 if base_seq is not None else 4
    cols = T
    
    precip_cmap, precip_norm = create_precipitation_cmap()
    
    obs_mm = obs_seq * MetricConfig.MM_MAX
    true_mm = true_seq * MetricConfig.MM_MAX
    pred_mm = pred_seq * MetricConfig.MM_MAX
    
    # 增加高度以容纳底部 Legend
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5 + 1.0))
    if T == 1: axes = axes[:, np.newaxis]
    
    fig.subplots_adjust(bottom=0.12, top=0.95, left=0.02, right=0.98, wspace=0.1, hspace=0.3)

    # 辅助函数：统一设置边框样式
    def setup_ax_border(ax, show_ylabel=False, ylabel_text=""):
        # 关键修改：不要使用 ax.axis('off')，否则边框会被隐藏
        # 而是隐藏刻度
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 显式开启边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(0.5)
            
        if show_ylabel:
            ax.set_ylabel(ylabel_text, fontsize=8)
            # 确保 Y 轴 Label 显示
            ax.yaxis.label.set_visible(True)

    input_len = obs_mm.shape[0]

    for t in range(T):
        # 1. Obs (Input)
        ax = axes[0, t]
        if t < input_len:
            ax.imshow(obs_mm[t], cmap=precip_cmap, norm=precip_norm)
            # [修改] 标签逻辑：倒序显示 T-x
            time_idx = input_len - 1 - t
            ax.set_title(f'T-{time_idx}', fontsize=6)
        else:
            ax.imshow(np.zeros_like(true_mm[0]), cmap=precip_cmap, norm=precip_norm)
        
        setup_ax_border(ax, show_ylabel=(t==0), ylabel_text='Obs')

        # 2. GT (Target)
        ax = axes[1, t]
        ax.imshow(true_mm[t], cmap=precip_cmap, norm=precip_norm)
        ax.set_title(f'T+{t+1}', fontsize=6)
        setup_ax_border(ax, show_ylabel=(t==0), ylabel_text='GT')

        # 3. Base Pred (Optional)
        current_row = 2
        if base_seq is not None:
            base_mm = base_seq * MetricConfig.MM_MAX
            ax = axes[current_row, t]
            ax.imshow(base_mm[t], cmap=precip_cmap, norm=precip_norm)
            setup_ax_border(ax, show_ylabel=(t==0), ylabel_text='Base')
            current_row += 1

        # 4. Prob Pred
        ax = axes[current_row, t]
        ax.imshow(pred_mm[t], cmap=precip_cmap, norm=precip_norm)
        setup_ax_border(ax, show_ylabel=(t==0), ylabel_text='Prob')
        current_row += 1
        
        # 5. Diff (GT - Prob Pred)
        ax = axes[current_row, t]
        diff = true_mm[t] - pred_mm[t]
        ax.imshow(diff, cmap='bwr', vmin=-30, vmax=30)
        setup_ax_border(ax, show_ylabel=(t==0), ylabel_text='Diff')

    # --- Legend 1: Precipitation (左侧) ---
    cbar_ax_precip = fig.add_axes([0.20, 0.05, 0.25, 0.015])
    sm_precip = plt.cm.ScalarMappable(cmap=precip_cmap, norm=precip_norm)
    sm_precip.set_array([])
    cbar_p = fig.colorbar(sm_precip, cax=cbar_ax_precip, orientation='horizontal', spacing='uniform')
    cbar_p.set_ticks([0.1, 1, 2, 5, 8])
    cbar_p.set_ticklabels(['0.1', '1', '2', '5', '8'])
    cbar_p.set_label('Precipitation (mm)', fontsize=8)
    cbar_p.ax.tick_params(labelsize=7)
    
    # --- Legend 2: Diff (右侧) ---
    cbar_ax_diff = fig.add_axes([0.55, 0.05, 0.25, 0.015])
    sm_diff = plt.cm.ScalarMappable(cmap='bwr', norm=plt.Normalize(vmin=-30, vmax=30))
    sm_diff.set_array([])
    cbar_d = fig.colorbar(sm_diff, cax=cbar_ax_diff, orientation='horizontal')
    cbar_d.set_ticks([-30, -15, 0, 15, 30])
    cbar_d.set_label('Difference (mm)', fontsize=8)
    cbar_d.ax.tick_params(labelsize=7)

    print(f'[INFO] Saving Plot to {out_path}')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

# ==========================================
# Part 4: 主入口函数 (Wrapper)
# ==========================================
def render(obs_seq, true_seq, pred_seq, base_seq, out_path: str, vmax: float = 1.0):
    # 1. 数据格式统一 (转 Numpy & 提取通道)
    def to_numpy_ch(x, ch=0):
        if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
        # 输入维度 (T, C, H, W), 通道 0 是雷达/降水标签
        if x.ndim == 4: x = x[:, ch] 
        return x

    obs = to_numpy_ch(obs_seq) # 取 ch=0 (Radar/Label)
    tru = to_numpy_ch(true_seq)
    prd = to_numpy_ch(pred_seq)
    
    # 处理基座预测
    base = to_numpy_ch(base_seq) if base_seq is not None else None
    
    print(f"Processing: {os.path.basename(out_path)}")
    
    # 2. 调用统计模块 (只计算 Prob 模型的指标)
    metrics_res = calc_seq_metrics(tru, prd, verbose=True)
    
    final_score = metrics_res['final_score']
    
    # 3. 调用绘图模块 (传入 base_seq)
    plot_seq_visualization(obs, tru, metrics_res['pred_clean'], base, metrics_res['score_per_frame'], out_path, vmax=vmax)
    
    return final_score

def parse_args():
    parser = argparse.ArgumentParser(description='Test SCWDS Probabilistic SimVP Model')
    parser.add_argument('--data_path', type=str, default='data/samples.jsonl')
    parser.add_argument('--in_shape', type=int, nargs=4, default=[10, 54, 256, 256],
                        help='Input shape (T, C, H, W). Default: [10, 54, 256, 256]')
    parser.add_argument('--aft_seq_length', type=int, default=20,
                        help='Output sequence length. Default: 20')
    parser.add_argument('--save_dir', type=str, default='./output/prob_simvp')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--accelerator', type=str, default='cuda')
    
    # [新增] 分箱参数
    parser.add_argument('--num_bins', type=int, default=64, help='概率分箱的数量')
    
    # [新增] 基座模型路径参数
    parser.add_argument('--base_ckpt_dir', type=str, default='./output/simvp', help='基座 SimVP 模型的 Checkpoint 目录')
    
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.accelerator if torch.cuda.is_available() else 'cpu')
    
    # 1. Config & Model
    config = SimVPConfig(
        data_path=args.data_path,
        in_shape=tuple(args.in_shape),
        save_dir=args.save_dir,
        aft_seq_length=args.aft_seq_length,
        out_channels=args.num_bins, # 设置 out_channels
        num_bins=args.num_bins       # 注入 num_bins
    )
    resize_shape = (config.in_shape[2], config.in_shape[3])
    
    # 加载 Prob 模型 Checkpoint
    ckpt_path = find_best_ckpt(config.save_dir)
    ckpt_info = get_checkpoint_info(ckpt_path)
    
    epoch = ckpt_info.get('epoch', 0)
    
    out_dir = os.path.join(config.save_dir, f'vis_prob_{epoch:02d}')
    os.makedirs(out_dir, exist_ok=True)
    
    log_file_path = os.path.join(out_dir, 'log.txt')
    set_logger(log_file_path)
    
    print(f"[INFO] Starting Probabilistic Test on {device}")
    print_checkpoint_info(ckpt_info)
    
    # [修改点 2] 使用 ProbabilisticSimVP 加载模型，并传入 num_bins 参数
    model = ProbabilisticSimVP.load_from_checkpoint(ckpt_path, num_bins=config.num_bins)
    model.eval().to(device)
    
    # [新增] 加载基座 SimVP 模型 (Regression)
    base_model = None
    if args.base_ckpt_dir:
        try:
            base_ckpt_path = find_best_ckpt(args.base_ckpt_dir)
            print(f"[INFO] Loading Base Model from: {base_ckpt_path}")
            base_model = SimVP_Regression.load_from_checkpoint(base_ckpt_path, map_location=device)
            base_model.eval().to(device)
        except Exception as e:
            print(f"[WARNING] 无法加载基座模型，将跳过基座对比: {e}")
            base_model = None

    # 2. Data
    data_module = ScwdsDataModule(
        data_path=config.data_path,
        resize_shape=resize_shape,
        batch_size=1,
        num_workers=4
    )
    data_module.setup('test')
    test_loader = data_module.test_dataloader()
    
    scores = []
    
    with torch.no_grad():
        for bidx, batch in enumerate(test_loader):
            # [关键修复] 将 Tensor 移动到正确的设备 (GPU)
            metadata, x, y, target_mask, input_mask = batch
            
            x = x.to(device)
            y = y.to(device)
            target_mask = target_mask.to(device)
            input_mask = input_mask.to(device)
            
            # 重新打包
            batch_on_device = (metadata, x, y, target_mask, input_mask)
            
            # 1. 运行 Prob 模型推理
            # ProbabilisticSimVP.test_step 已执行 Argmax 解码和归一化
            outputs = model.test_step(batch_on_device, bidx)
            
            # 2. [新增] 运行 Base 模型推理
            base_preds = None
            if base_model is not None:
                # SimVP forward 返回 raw logits [B, T, C, H, W]
                
                # [🔥 关键修复] 显式调用 Resize，将 301x301 -> 256x256
                # 否则模型内部 Skip Connection 会出现 304 vs 301 的尺寸不匹配
                x_base = base_model._interpolate_batch_gpu(x, mode='max_pool')
                
                base_logits = base_model(x_base)
                base_sigmoid = torch.sigmoid(base_logits)
                base_clamped = torch.clamp(base_sigmoid, 0.0, 1.0)
                # 提取 numpy 数据 [T, C, H, W]
                base_preds = base_clamped[0].cpu().numpy()
            
            save_path = os.path.join(out_dir, f'sample_{bidx:03d}.png')
            
            # outputs['preds'] 是 Argmax 解码后的归一化数值 [T, 1, H, W]
            # 传入 base_preds 进行对比渲染
            s = render(outputs['inputs'], outputs['trues'], outputs['preds'], base_preds, save_path)
            scores.append(s)
            
            if bidx >= args.num_samples - 1:
                break
    
    if len(scores) > 0:
        print(f"\n[FINAL] Average Probabilistic Score ({len(scores)} samples): {np.mean(scores):.6f}")
    
    restore_stdout()

if __name__ == '__main__':
    try:
        main()
    finally:
        restore_stdout()