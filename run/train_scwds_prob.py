# train_scwds_prob.py (概率分箱 SimVP-Mamba 迁移学习训练脚本)

import sys
import os
import glob
from datetime import datetime
import argparse
import ast
from pydantic import ValidationError

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import lightning as l
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from metai.dataset.met_dataloader_scwds import ScwdsDataModule
from metai.model.simvp.simvp_config import SimVPConfig
# 导入 Probabilistic Trainer 和 SimVP 基座模型（用于加载权重）
from metai.model.simvp.prob_trainer import ProbabilisticSimVP # 假设ProbabilisticSimVP位于此
from metai.model.simvp.simvp_trainer import SimVP # 假设SimVP (回归版)位于此
from metai.utils import MLOGI

def find_best_ckpt(save_dir: str) -> str:
    """查找最优或最新的 Checkpoint 文件，优先 best.ckpt"""
    # 优先查找 best.ckpt
    best = os.path.join(save_dir, 'best.ckpt')
    if os.path.exists(best): return best
    
    # 其次查找 last.ckpt
    last = os.path.join(save_dir, 'last.ckpt')
    if os.path.exists(last): return last
    
    # 最后查找所有 checkpoint 文件，返回最新的
    cpts = sorted(glob.glob(os.path.join(save_dir, '*.ckpt')))
    if len(cpts) > 0: return cpts[-1]
        
    raise FileNotFoundError(f'No checkpoint found in {save_dir}')

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train SCWDS Probabilistic SimVP Model (Transfer Learning)')
    
    # 基础路径与数据参数
    parser.add_argument('--data_path', type=str, default='data/samples.jsonl', help='Path to training data')
    parser.add_argument('--save_dir', type=str, default='./output/prob_simvp', help='Output directory for Probabilistic Model')
    parser.add_argument('--in_shape', type=int, nargs=4, default=None) 
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--max_epochs', type=int, default=30, help='微调的最大训练轮数')
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--aft_seq_length', type=int, default=None)

    # [核心参数] 概率分箱参数
    parser.add_argument('--num_bins', type=int, default=64, help='概率分箱的数量')
    
    # [迁移学习参数]
    parser.add_argument('--base_ckpt_dir', type=str, required=True, help='SimVP基座模型(回归版)的保存目录，将自动查找 best.ckpt')
    parser.add_argument('--ckpt_path', type=str, default=None, help='如果指定，则直接加载该路径的模型作为初始权重')
    
    # 模型结构参数
    parser.add_argument('--model_type', type=str, default=None)
    parser.add_argument('--hid_S', type=int, default=None)
    parser.add_argument('--hid_T', type=int, default=None)
    parser.add_argument('--N_S', type=int, default=None)
    parser.add_argument('--N_T', type=int, default=None)
    parser.add_argument('--mlp_ratio', type=float, default=None)
    parser.add_argument('--drop', type=float, default=None)
    parser.add_argument('--drop_path', type=float, default=None)
    
    # 优化器
    parser.add_argument('--opt', type=str, default='adamw')
    parser.add_argument('--lr', type=float, default=1e-4, help='微调阶段的推荐学习率 (1e-4 ~ 1e-5)')
    parser.add_argument('--sched', type=str, default='cosine')
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--warmup_epoch', type=int, default=2)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--gradient_clip_algorithm', type=str, default='norm')
    
    # 设备与精度
    parser.add_argument('--accelerator', type=str, default='cuda')
    parser.add_argument('--devices', type=str, default='auto')
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    
    # 早停参数
    parser.add_argument('--early_stop_patience', type=int, default=10)
    parser.add_argument('--early_stop_monitor', type=str, default='val_score')
    parser.add_argument('--early_stop_mode', type=str, default='max')

    return parser.parse_args()

def main():
    torch.set_float32_matmul_precision('high')
    args = parse_args()
    
    config_kwargs = {k: v for k, v in vars(args).items() if v is not None}
    
    if 'in_shape' in config_kwargs: config_kwargs['in_shape'] = tuple(config_kwargs['in_shape'])
    
    # [核心设置] 强制设置 out_channels 为 num_bins
    num_bins = config_kwargs.get('num_bins', 64)
    config_kwargs['out_channels'] = num_bins 
    
    # ... (设备和布尔值处理，保持与 SimVP 脚本一致) ...

    try:
        config = SimVPConfig(**config_kwargs)
    except ValidationError as e:
        MLOGI(f"[ERROR] Config Validation: {e}")
        return

    l.seed_everything(config.seed)

    data_module = ScwdsDataModule(
        data_path=config.data_path,
        resize_shape=config.resize_shape,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_split=config.train_split,
        val_split=config.val_split,
        test_split=config.test_split,
        seed=config.seed
    )
    
    # 1. 初始化概率分箱模型
    model_args = config.to_dict()
    model = ProbabilisticSimVP(**model_args)

    # 2. [核心] 查找并加载基座模型权重 (Transfer Learning)
    base_ckpt_path = args.ckpt_path
    if base_ckpt_path is None:
        try:
            base_ckpt_path = find_best_ckpt(args.base_ckpt_dir)
        except FileNotFoundError:
            MLOGI(f"[WARNING] 未在 {args.base_ckpt_dir} 找到 Checkpoint，将从头开始训练。")
            base_ckpt_path = None
    
    if base_ckpt_path:
        MLOGI(f"🚀 启用迁移学习: 载入 SimVP 基座权重自: {base_ckpt_path}")
        try:
            ckpt = torch.load(base_ckpt_path, map_location='cpu')
            state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            
            new_state_dict = {}
            for k, v in state_dict.items():
                # 仅跳过 readout 层，其他层均加载
                if 'model.readout' in k: 
                    continue
                new_state_dict[k] = v

            # 载入除 readout 之外的所有权重 (strict=False 允许 readout 层缺失)
            model.load_state_dict(new_state_dict, strict=False)
            MLOGI("[INFO] Backbone (Encoder+Mamba+Decoder) 权重加载成功。Readout 层将从随机初始化开始学习。")

        except Exception as e:
            MLOGI(f"[ERROR] 加载基座模型权重失败: {e}。将从随机初始化开始训练。")

    # 3. Callbacks 和 Trainer 初始化
    monitor_metric = config.early_stop_monitor
    monitor_mode = config.early_stop_mode

    callbacks = [
        # 早停
        EarlyStopping(
            monitor=monitor_metric, 
            min_delta=config.early_stop_min_delta, 
            patience=config.early_stop_patience, 
            mode=monitor_mode, 
            verbose=True
        ),
        
        # 保存最优模型
        ModelCheckpoint(
            dirpath=config.save_dir, 
            filename="prob-{epoch:02d}-{val_score:.4f}",
            monitor=monitor_metric,
            save_top_k=3, 
            mode=monitor_mode,
            save_last=True # 总是保存 last.ckpt 用于断点续训
        ),
        
        LearningRateMonitor(logging_interval="step")
    ]

    logger = TensorBoardLogger(save_dir=config.save_dir, name=config.model_name, version=datetime.now().strftime("%Y%m%d-%H%M%S"))

    # DDP Strategy
    strategy = 'ddp_find_unused_parameters_false' if config.devices != 1 and config.accelerator == 'cuda' else 'auto'

    trainer = l.Trainer(
        max_epochs=config.max_epochs,
        default_root_dir=config.save_dir,
        precision=config.precision,
        accelerator=config.accelerator,
        devices=config.devices,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config.log_every_n_steps,
        val_check_interval=config.val_check_interval,
        gradient_clip_val=config.gradient_clip_val,
        gradient_clip_algorithm=config.gradient_clip_algorithm,
        accumulate_grad_batches=config.accumulate_grad_batches,
        strategy=strategy,
        sync_batchnorm=False, 
        enable_progress_bar=config.enable_progress_bar,
        enable_model_summary=config.enable_model_summary,
        num_sanity_val_steps=config.num_sanity_val_steps,
    )

    MLOGI(f"Starting Probabilistic Training: Model={config.model_type}, Bins={config.num_bins}")
    MLOGI(f"  Transfer Learning Source: {base_ckpt_path or 'None'}")
    
    # 4. 启动训练
    trainer.fit(model, datamodule=data_module, ckpt_path=args.ckpt_path)

if __name__ == "__main__":
    main()