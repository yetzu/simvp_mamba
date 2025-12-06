# run/train_scwds_prob.py
# ==============================================================================
# 功能: 概率分箱 SimVP-Mamba 迁移学习训练脚本 (Probabilistic Transfer Learning)
# 特性:
#   1. 集成 Focal Loss + Gaussian Soft Label 以解决序数丢失和虚警问题。
#   2. 实现两阶段微调 (Two-Stage Finetuning): 先冻结 Backbone 训练 Head，再全网微调。
#   3. 支持自动查找基座模型 Checkpoint 进行热启动。
# ==============================================================================

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
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, BaseFinetuning

from metai.dataset.met_dataloader_scwds import ScwdsDataModule
from metai.model.simvp.simvp_config import SimVPConfig
from metai.model.simvp.prob_trainer import ProbabilisticSimVP 
from metai.utils import MLOGI

# ==============================================================================
# 自定义回调：概率模型微调策略 (Probabilistic Finetuning Callback)
# ==============================================================================
class ProbabilisticFinetuning(BaseFinetuning):
    """
    实现“先冻结后解冻”的迁移学习策略。
    
    阶段 1 (Warmup): 
        - 冻结 SimVP Backbone (Encoder, Translator/Mamba, Decoder)。
        - 仅训练 Readout 层 (C_hid -> num_bins)，使其适应分类任务的输出分布。
        
    阶段 2 (Finetuning):
        - 在 `unfreeze_at_epoch` 轮次解冻所有层。
        - 进行全参数微调，优化整体特征提取能力。
    """
    def __init__(self, unfreeze_at_epoch=2):
        super().__init__()
        self.unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module):
        # 冻结除 readout 外的所有层
        # 注意：train_bn=False 意味着 BN 层的统计量(running_mean/var)不会更新，但 gamma/beta 会被冻结
        self.freeze(pl_module.model.enc, train_bn=False)
        self.freeze(pl_module.model.hid, train_bn=False)
        self.freeze(pl_module.model.dec, train_bn=False)
        
        # 确保 readout 是解冻的 (这是我们要从头训练的层)
        self.make_trainable(pl_module.model.readout)
        MLOGI("🥶 [Finetuning] Backbone frozen for warmup. Training only Readout layer.")

    def finetune_function(self, pl_module, current_epoch, optimizer):
        # 在指定 epoch 解冻
        if current_epoch == self.unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=pl_module.model,
                optimizer=optimizer,
                train_bn=True, # 解冻后允许 BN 更新
            )
            MLOGI(f"🔥 [Finetuning] Backbone unfrozen at epoch {current_epoch}. Full finetuning started.")

# ==============================================================================
# 辅助函数
# ==============================================================================
def find_best_ckpt(save_dir: str) -> str:
    """查找最优或最新的 Checkpoint 文件，优先 best.ckpt"""
    best = os.path.join(save_dir, 'best.ckpt')
    if os.path.exists(best): return best
    
    last = os.path.join(save_dir, 'last.ckpt')
    if os.path.exists(last): return last
    
    cpts = sorted(glob.glob(os.path.join(save_dir, '*.ckpt')))
    if len(cpts) > 0: return cpts[-1]
        
    raise FileNotFoundError(f'No checkpoint found in {save_dir}')

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train SCWDS Probabilistic SimVP Model (Transfer Learning)')
    
    # --- 基础路径与数据参数 ---
    parser.add_argument('--data_path', type=str, default='data/samples.jsonl', help='Path to training data')
    parser.add_argument('--save_dir', type=str, default='./output/prob_simvp', help='Output directory')
    parser.add_argument('--in_shape', type=int, nargs=4, default=None) 
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--max_epochs', type=int, default=30, help='Total training epochs')
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--aft_seq_length', type=int, default=None)

    # --- [核心改进] 概率分箱与 Loss 参数 ---
    parser.add_argument('--num_bins', type=int, default=40, help='概率分箱数 (建议 40)')
    parser.add_argument('--sigma', type=float, default=2.0, help='Soft Label 高斯标准差 (建议 2.0)')
    parser.add_argument('--use_focal', type=str, default='true', help='启用 Focal Loss (true/false)')
    parser.add_argument('--gamma', type=float, default=2.0, help='Focal Loss 聚焦参数')

    # --- [迁移学习参数] ---
    parser.add_argument('--base_ckpt_dir', type=str, required=True, help='SimVP基座模型目录 (用于加载权重)')
    parser.add_argument('--ckpt_path', type=str, default=None, help='指定加载的 Checkpoint 路径 (Resume)')
    parser.add_argument('--unfreeze_epoch', type=int, default=3, help='解冻 Backbone 的 Epoch (Warmup 轮数)')
    
    # --- 模型结构参数 ---
    parser.add_argument('--model_type', type=str, default=None)
    parser.add_argument('--hid_S', type=int, default=None)
    parser.add_argument('--hid_T', type=int, default=None)
    parser.add_argument('--N_S', type=int, default=None)
    parser.add_argument('--N_T', type=int, default=None)
    parser.add_argument('--mlp_ratio', type=float, default=None)
    parser.add_argument('--drop', type=float, default=None)
    parser.add_argument('--drop_path', type=float, default=None)
    
    # --- 优化器 ---
    parser.add_argument('--opt', type=str, default='adamw')
    parser.add_argument('--lr', type=float, default=2e-4, help='初始学习率 (建议 2e-4)')
    parser.add_argument('--sched', type=str, default='cosine')
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--warmup_epoch', type=int, default=0, help='LR Warmup (注意与 Backbone Warmup 区分)')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    
    # --- 设备与精度 ---
    parser.add_argument('--accelerator', type=str, default='cuda')
    parser.add_argument('--devices', type=str, default='auto')
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    
    # --- 早停 ---
    parser.add_argument('--early_stop_patience', type=int, default=10)
    parser.add_argument('--early_stop_monitor', type=str, default='val_score')
    parser.add_argument('--early_stop_mode', type=str, default='max')

    return parser.parse_args()

# ==============================================================================
# 主程序
# ==============================================================================
def main():
    torch.set_float32_matmul_precision('high')
    args = parse_args()
    
    # 1. 参数预处理
    config_kwargs = {k: v for k, v in vars(args).items() if v is not None}
    
    if 'in_shape' in config_kwargs: config_kwargs['in_shape'] = tuple(config_kwargs['in_shape'])
    
    # 布尔值解析
    if isinstance(config_kwargs.get('use_focal'), str):
        config_kwargs['use_focal'] = config_kwargs['use_focal'].lower() == 'true'

    # 强制同步 num_bins 到 out_channels
    num_bins = config_kwargs.get('num_bins', 40)
    config_kwargs['out_channels'] = num_bins 
    
    # 2. 初始化 Config
    try:
        # 移除 Config 类不接受的额外参数 (如 sigma, use_focal, unfreeze_epoch 等)
        valid_keys = SimVPConfig.model_fields.keys()
        safe_kwargs = {k: v for k, v in config_kwargs.items() if k in valid_keys}
        
        config = SimVPConfig(**safe_kwargs)
    except ValidationError as e:
        MLOGI(f"[ERROR] Config Validation: {e}")
        return

    l.seed_everything(config.seed)

    # 3. 初始化 DataModule
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
    
    # 4. 初始化模型 (手动注入 Loss 参数)
    model_args = config.to_dict()
    
    # [关键] 注入 ProbabilisticSimVP 所需的特定参数
    model_args['num_bins'] = num_bins
    model_args['sigma'] = config_kwargs.get('sigma', 2.0)
    model_args['use_focal'] = config_kwargs.get('use_focal', True)
    model_args['gamma'] = config_kwargs.get('gamma', 2.0)
    
    MLOGI(f"[Init] Model: Bins={num_bins}, Sigma={model_args['sigma']}, Focal={model_args['use_focal']}")
    
    model = ProbabilisticSimVP(**model_args)

    # 5. 迁移学习：加载基座权重 (Backbone Loading)
    base_ckpt_path = args.ckpt_path # 如果指定了特定 ckpt，则优先使用
    
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
                # 过滤掉 Readout 层 (因为输出通道数不匹配: 1 vs 40)
                if 'model.readout' in k: 
                    continue
                new_state_dict[k] = v

            # 载入 Backbone 权重 (strict=False 允许缺失 readout)
            model.load_state_dict(new_state_dict, strict=False)
            MLOGI("[INFO] Backbone 权重加载成功。Readout 层将从随机初始化开始学习。")
            
        except Exception as e:
            MLOGI(f"[ERROR] 加载基座模型权重失败: {e}。将从随机初始化开始训练。")

    # 6. 配置 Callbacks
    monitor_metric = config.early_stop_monitor
    monitor_mode = config.early_stop_mode

    callbacks = [
        # [关键] 迁移学习微调策略
        ProbabilisticFinetuning(unfreeze_at_epoch=args.unfreeze_epoch),
        
        # 早停策略
        EarlyStopping(
            monitor=monitor_metric, 
            min_delta=config.early_stop_min_delta, 
            patience=config.early_stop_patience, 
            mode=monitor_mode, 
            verbose=True
        ),
        
        # 权重保存
        ModelCheckpoint(
            dirpath=config.save_dir, 
            filename="prob-{epoch:02d}-{val_score:.4f}",
            monitor=monitor_metric,
            save_top_k=3, 
            mode=monitor_mode,
            save_last=True 
        ),
        
        LearningRateMonitor(logging_interval="step")
    ]

    logger = TensorBoardLogger(save_dir=config.save_dir, name=config.model_name, version=datetime.now().strftime("%Y%m%d-%H%M%S"))

    # DDP 策略配置
    strategy = 'ddp_find_unused_parameters_false' if config.devices != 1 and config.accelerator == 'cuda' else 'auto'

    # 7. 初始化 Trainer
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
        strategy=strategy,
        sync_batchnorm=False, 
        enable_progress_bar=config.enable_progress_bar,
        enable_model_summary=config.enable_model_summary,
        num_sanity_val_steps=config.num_sanity_val_steps,
    )

    MLOGI(f"Starting Training with Unfreeze Epoch: {args.unfreeze_epoch}")
    
    # 8. 启动训练
    # 注意：如果 args.ckpt_path 被指定且是为了 Resume (而非迁移学习), 这里应该传给 ckpt_path 参数
    # 但根据当前逻辑，args.ckpt_path 用于迁移学习加载，所以 Trainer.fit 不传 ckpt_path (从头开始 epoch 计数)
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()