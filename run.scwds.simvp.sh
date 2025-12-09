#!/bin/bash

# SimVP SCWDS 全流程脚本 (Optimized for 4x A800 80GB)
# 包含: Train (SimVP) -> Test (SimVP) -> Infer (SimVP) -> [NEW] Probabilistic Model
# Usage: bash run.scwds.simvp.sh [MODE]

# ================= 环境变量优化 =================
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN

# ================= 参数检查 =================
if [ $# -eq 0 ]; then
    echo "错误: 请指定操作模式"
    echo "用法: bash run.scwds.simvp.sh [MODE]"
    echo "支持的模式:"
    echo " train      - 训练 SimVP 基座模型"
    echo " test       - 测试 SimVP 基座模型"
    echo " infer      - 使用 SimVP 基座进行推理"
    echo " infer_gpm  - 使用 Soft-GPM 后处理推理"
    exit 1
fi

MODE=$1

case $MODE in
    # ============================================================
    # 1. 训练 SimVP 基座 (Stage 1) - [保持原样]
    # ============================================================
    "train")
        echo "--------------------------------------------------------"
        echo "🚀 开始训练 Mamba 基座模型 (BF16 Mixed)..."
        echo "--------------------------------------------------------"
        
        python run/train_scwds_simvp.py \
            --data_path data/samples.jsonl \
            --save_dir ./output/simvp \
            --batch_size 3 \
            --accumulate_grad_batches 4 \
            --num_workers 8 \
            \
            --in_shape 10 54 256 256 \
            --aft_seq_length 20 \
            --max_epochs 100 \
            --opt adamw \
            --lr 5e-4 \
            --sched cosine \
            --min_lr 1e-6 \
            --warmup_epoch 5 \
            \
            --model_type mamba \
            --hid_S 128 \
            --hid_T 1024 \
            --N_S 4 \
            --N_T 16 \
            --mlp_ratio 4.0 \
            --drop 0.05 \
            --drop_path 0.3 \
            --spatio_kernel_enc 7 \
            --spatio_kernel_dec 7 \
            --loss_weight_l1 1.0 \
            --loss_weight_csi 1.0 \
            --loss_weight_ssim 0.5 \
            --loss_weight_evo 0.5 \
            --loss_weight_spectral 0.1 \
            \
            --use_curriculum_learning false \
            --early_stop_patience 15 \
            --early_stop_monitor val_score \
            --early_stop_mode max \
            --accelerator cuda \
            --devices 1,2,3 \
            --precision bf16-mixed \
            --gradient_clip_val 0.5 \
            --gradient_clip_algorithm norm
        ;;
        
    # ============================================================
    # 2. 测试 SimVP 基座
    # ============================================================
    "test")
        echo "----------------------------------------"
        echo "🧪 开始测试 Mamba 基座模型..."
        echo "----------------------------------------"
        
        python run/test_scwds_simvp.py \
            --data_path data/samples.jsonl \
            --in_shape 10 54 256 256 \
            --aft_seq_length 20 \
            --save_dir ./output/simvp \
            --num_samples 10 \
            --accelerator cuda
        ;;
        
    # ============================================================
    # 3. 推理 SimVP 基座 - [保持原样]
    # ============================================================
    "infer")
        echo "----------------------------------------"
        echo "🔮 开始推理 Mamba 模型..."
        echo "----------------------------------------"
        
        python run/infer_scwds_simvp.py \
            --data_path data/samples.testset.jsonl \
            --in_shape 20 54 256 256 \
            --save_dir ./output/simvp \
            --accelerator cuda:0 \
            --vis
        ;;

    # ============================================================
    # 4. 推理 SimVP 基座 + Soft-GPM 后处理
    # ============================================================
    "infer_gpm")
        echo "----------------------------------------"
        echo "🔮 开始推理 SimVP (Soft-GPM) 模型..."
        echo "----------------------------------------"
        
        python run/infer_scwds_simvp_gpm.py \
            --data_path data/samples.testset.jsonl \
            --in_shape 20 54 256 256 \
            --save_dir ./output/simvp \
            --accelerator cuda:0 \
            --vis \
            --vis_output ./output/simvp/vis_infer_gpm \
            --gpm_alpha 0.5 \
            --gpm_decay 0.98
        ;;
        
esac

echo "✅ 操作完成！"