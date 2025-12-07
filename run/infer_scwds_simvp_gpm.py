# run/infer_scwds_simvp_gpm.py
# ==============================================================================
# 功能: 使用训练好的 SimVP 模型进行推理，并应用 Soft-GPM (Gaussian Probability Matching)
#       策略对预测结果进行后处理，以恢复降水极值分布。
# 特性: 包含完整的容错机制，确保在 GPM 计算失败时自动降级为普通推理，保证文件生成的完整性。
# ==============================================================================

import sys
import os
import glob
import argparse
import traceback
from datetime import datetime, timedelta

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# 设置 matplotlib 后端为 Agg，适用于无头服务器环境
matplotlib.use('Agg')

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metai.utils import MetLabel, MLOGI, MLOGE
from metai.dataset.met_dataloader_scwds import ScwdsDataModule
from metai.model.simvp import SimVPConfig, SimVP
from metai.utils.met_config import get_config

# ==========================
# 竞赛常量定义
# ==========================
USER_ID = "CP2025000081"
TRACK_ID = "track"
TIME_STEP_MINUTES = 6 

# ==========================
# GPM 后处理工具类
# ==========================
class PostProcessor:
    """
    后处理工具类，包含概率匹配 (GPM) 等算法。
    """
    
    @staticmethod
    def probability_matching(pred_img: np.ndarray, ref_source: np.ndarray) -> np.ndarray:
        """
        标准 GPM (Probability Matching) 计算。
        将预测图像 (pred_img) 的直方图分布强制调整为参考图像 (ref_source) 的分布。
        
        Args:
            pred_img: 当前预测帧 (H, W)
            ref_source: 参考源图像 (H, W)，通常取最近时刻的观测值
            
        Returns:
            matched_img: 分布匹配后的图像 (H, W)
        """
        # 展平数组以便排序
        pred_flat = pred_img.flatten()
        ref_flat = ref_source.flatten() 
        
        # 获取预测值的排序索引
        pred_indices = np.argsort(pred_flat)
        
        # 对参考源进行排序
        ref_sorted = np.sort(ref_flat)
        
        # 处理数量不一致的情况 (线性插值对齐)
        # 确保参考源的像素数量与预测图一致
        if len(ref_sorted) != len(pred_flat):
            ref_sorted = np.interp(
                np.linspace(0, 1, len(pred_flat)),
                np.linspace(0, 1, len(ref_sorted)),
                ref_sorted
            )
            
        # 构造匹配后的数组
        matched_flat = np.zeros_like(pred_flat)
        # 将排序后的参考值，按预测值原本的大小顺序填回
        matched_flat[pred_indices] = ref_sorted
        
        return matched_flat.reshape(pred_img.shape)

# ==========================
# 辅助函数
# ==========================
def find_latest_ckpt(save_dir: str) -> str:
    """查找目录下最新的 checkpoint 文件，优先 best.ckpt"""
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
    
    # 统一归一化显示
    vmax = obs_seq.max() if obs_seq.max() > 1 else 1.0
    
    for t in range(cols):
        # 绘制观测 (Input)
        ax = axes[0, t]
        if t < T_in:
            ax.imshow(obs_seq[t], cmap='turbo', vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if t == 0: ax.set_title('Obs (Past)', fontsize=10)
        else: ax.axis('off')

        # 绘制预测 (Pred)
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
    parser = argparse.ArgumentParser(description='Infer SCWDS SimVP Model with Soft-GPM')
    
    # 基础参数
    parser.add_argument('--data_path', type=str, default='data/samples.testset.jsonl')
    parser.add_argument('--in_shape', type=int, nargs=4, default=[20, 11, 256, 256])
    parser.add_argument('--save_dir', type=str, default='./output/simvp')
    parser.add_argument('--accelerator', type=str, default='cuda')
    parser.add_argument('--vis', action='store_true', help='Enable visualization')
    parser.add_argument('--vis_output', type=str, default='./output/simvp/vis_infer_gpm')
    
    # Soft-GPM 核心参数
    parser.add_argument('--gpm_alpha', type=float, default=0.5, 
                        help='GPM 融合初始权重 (0.0=原模型, 1.0=纯GPM, 推荐 0.5)')
    parser.add_argument('--gpm_decay', type=float, default=0.98, 
                        help='GPM 权重随时间的衰减系数 (推荐 0.95-0.99)')
    
    return parser.parse_args()

# ==========================
# 主程序
# ==========================
def main():
    args = parse_args()
    met_config = get_config() 
    
    # 1. 初始化配置
    config = SimVPConfig(
        data_path=args.data_path,
        in_shape=tuple(args.in_shape),
        save_dir=args.save_dir,
    )
    resize_shape = (config.in_shape[2], config.in_shape[3])
    device = torch.device(args.accelerator if torch.cuda.is_available() else 'cpu')
    FMT = met_config.file_date_format
    
    # 物理最大值 (30.0 mm)，用于归一化换算
    PHYSICAL_MAX_MM = MetLabel.RA.max / 10.0 
    
    MLOGI(f"[INFO] Start Soft-GPM Inference. Device: {device}")
    MLOGI(f"[INFO] Config: Alpha={args.gpm_alpha}, Decay={args.gpm_decay}")

    # 2. 准备数据加载器
    data_module = ScwdsDataModule(
        data_path=config.data_path,
        resize_shape=resize_shape,
        batch_size=1,
        num_workers=1
    )
    infer_loader = data_module.infer_dataloader()
    
    # 3. 加载模型
    try:
        ckpt_path = find_latest_ckpt(config.save_dir)
        MLOGI(f"加载检查点: {ckpt_path}")
        model = SimVP.load_from_checkpoint(ckpt_path, map_location=device)
        model.eval().to(device)
    except Exception as e:
        MLOGE(f"模型加载失败: {e}")
        return

    # 4. 推理循环
    with torch.no_grad():
        # 注意：DataModule 的 infer_dataloader 返回的是 (metadata, input, mask)
        for bidx, (metadata_list, batch_x, input_mask) in enumerate(infer_loader):
            # 获取当前 Batch 的元数据（Batch Size = 1）
            metadata = metadata_list[0]
            sample_id = metadata['sample_id']
            
            try:
                batch_x = batch_x.to(device)
                
                # --- A. 模型推理 ---
                # batch_y: [B, T, C, H, W] -> [1, 20, 1, 256, 256]
                batch_y = model.infer_step((metadata_list, batch_x, input_mask), batch_idx=bidx)
                # 去除 B 和 C 维度 -> [20, 256, 256]
                batch_y = batch_y[0, :, 0, :, :] 
                
                # --- B. 解析元数据 ---
                case_id = metadata.get('case_id')
                sample_id_parts = sample_id.split('_')
                task_id = metadata.get('task_id')
                region_id = metadata['region_id']
                station_id = metadata['station_id']
                time_id = sample_id_parts[2] 
                timestamps = metadata['timestamps']
                
                # 检查时间戳有效性
                if not timestamps:
                    MLOGE(f"样本 {sample_id} 缺少时间戳，跳过")
                    continue
                
                # 计算起始时间
                last_obs_idx = batch_x.shape[1] - 1
                if last_obs_idx >= len(timestamps):
                    last_obs_idx = -1
                
                last_obs_dt = datetime.strptime(timestamps[last_obs_idx], FMT)

                # =========================================================
                # [Soft-GPM 策略] 包含容错回退机制
                # =========================================================
                base_apply_gpm = False
                ref_source = None

                # [容错块 1] 尝试准备 GPM 参考源
                try:
                    # 检查维度：确保 batch_x 有足够的通道维度进行切片
                    # 预期 batch_x 形状: [1, T_in, C_in, H, W]
                    if batch_x.dim() == 5 and batch_x.shape[2] > 0:
                        # 提取最近时刻的观测作为参考源 (第 0 通道通常是雷达/降水)
                        input_seq_tensor = batch_x[0, :, 0, :, :] # [T, H, W]
                        # 简单地取最近一帧，或者取时间维度的最大值作为参考
                        # 这里取最近一帧 (Last Observation)
                        ref_source = input_seq_tensor[-1].cpu().numpy()
                        
                        # 强度门控：如果参考源最大值太小，则不启用 GPM (避免放大噪声)
                        ref_max_norm = ref_source.max()
                        THRESHOLD_MM_PHYSICAL = 1.0 # 物理阈值 1.0mm
                        DISABLE_GPM_THRESHOLD = THRESHOLD_MM_PHYSICAL / PHYSICAL_MAX_MM
                        
                        if ref_max_norm >= DISABLE_GPM_THRESHOLD:
                            base_apply_gpm = True
                    else:
                        MLOGE(f"样本 {sample_id} 维度异常 {batch_x.shape}，无法准备 GPM 参考源")
                except Exception as gpm_prep_err:
                    MLOGE(f"样本 {sample_id} GPM 准备阶段失败: {gpm_prep_err}，将使用普通推理")
                    base_apply_gpm = False

                pred_frames_vis = []
                seq_mean_val_stored = 0.0
                
                # --- C. 逐帧处理与保存 ---
                for idx, y in enumerate(batch_y):
                    # 1. 插值原始预测到目标尺寸 (301x301)
                    # 输入 y 是 (256, 256)，需要 unsqueeze 到 (1, 1, 256, 256) 进行插值
                    y_interp = F.interpolate(
                        y.unsqueeze(0).unsqueeze(0), 
                        size=(301, 301),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze() # -> (301, 301)
                    
                    y_raw_np = y_interp.cpu().numpy() # 原始预测 (Raw Prediction)

                    # 2. 计算 GPM (带容错)
                    y_final_np = y_raw_np # 默认回退到原始预测
                    
                    if base_apply_gpm:
                        try:
                            # 2.1 生成全量 GPM 匹配结果
                            # 注意：ref_source 也需要插值到 301x301 才能匹配
                            # 为节省性能，这里我们在 GPM 内部做隐式匹配，或者先插值 ref
                            # 简单起见，这里假设 PostProcessor 能处理 (ref_source 需要resize)
                            
                            # 临时将 ref_source resize 到 301x301 以匹配 y_raw_np
                            # (仅在第一次计算时需要，这里为简单每次都做)
                            ref_source_resized = F.interpolate(
                                torch.from_numpy(ref_source).unsqueeze(0).unsqueeze(0),
                                size=(301, 301), mode='nearest'
                            ).squeeze().numpy()

                            y_gpm_np = PostProcessor.probability_matching(y_raw_np, ref_source_resized)
                            
                            # 2.2 计算当前帧的融合权重 (随时间衰减)
                            # t=0: alpha * decay^0 = alpha
                            # t=19: alpha * decay^19
                            current_alpha = args.gpm_alpha * (args.gpm_decay ** idx)
                            
                            # 2.3 线性融合 (Soft-GPM)
                            y_final_np = (1.0 - current_alpha) * y_raw_np + current_alpha * y_gpm_np
                            
                        except Exception as gpm_err:
                            # GPM 计算出错，静默降级为使用 y_raw_np
                            # 仅在第一帧打印错误日志，避免刷屏
                            if idx == 0: 
                                MLOGE(f"样本 {sample_id} GPM 计算出错: {gpm_err}，已降级")
                            y_final_np = y_raw_np

                    # 3. 阈值清理 (去除微小底噪)
                    y_final_np[y_final_np < 0.003] = 0.0
                    
                    # 4. 还原为存储数值 (0-300) 并转为 float32
                    y_save = (y_final_np * MetLabel.RA.max).astype(np.float32)
                    
                    # 5. 构造保存路径
                    forecast_dt = last_obs_dt + timedelta(minutes=TIME_STEP_MINUTES * (idx + 1))
                    forecast_time_str = forecast_dt.strftime(FMT)
                    
                    npy_dir = os.path.join('submit', 'output', USER_ID, TRACK_ID, case_id)
                    os.makedirs(npy_dir, exist_ok=True)
                    
                    npy_path = os.path.join(
                        npy_dir,
                        f"{task_id}_{region_id}_{time_id}_{station_id}_Forcast_{forecast_time_str}.npy"
                    )
                    
                    # 6. 保存文件
                    np.save(npy_path, y_save)
                    
                    seq_mean_val_stored += float(y_save.mean())
                    
                    # 收集可视化数据 (用于绘图)
                    if args.vis:
                        pred_frames_vis.append(y_save / 300.0)
                
                # --- D. 打印日志 ---
                seq_mean_val_stored /= len(batch_y)
                gpm_status = f"Soft(A={args.gpm_alpha})" if base_apply_gpm else "OFF(Safe)"
                MLOGI(f"No.{bidx} {sample_id} | {gpm_status} | Mean: {seq_mean_val_stored/10.0:.4f}mm")

                # --- E. 可视化 (可选) ---
                if args.vis:
                    obs_frames = batch_x[0, :, 0, :, :].cpu().numpy()
                    pred_frames = np.array(pred_frames_vis)
                    vis_path = os.path.join(args.vis_output, f"{sample_id}.png")
                    plot_inference(obs_frames, pred_frames, vis_path)

            except Exception as e:
                # 捕获整个样本处理过程中的其他未知错误，确保循环不中断
                traceback.print_exc()
                MLOGE(f"样本 {bidx} ({sample_id}) 发生严重未捕获异常: {e}")
                continue
            
    MLOGI("✅ Soft-GPM 推理流程全部完成！")

if __name__ == '__main__':
    main()