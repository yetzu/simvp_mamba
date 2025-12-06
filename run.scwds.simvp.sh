#!/bin/bash

# SimVP SCWDS 全流程脚本 (Optimized for 4x A800 80GB)
# 包含: Train (SimVP) -> Test (SimVP) -> Infer (SimVP) -> [NEW] Probabilistic Model
# Usage: bash run.scwds.simvp.sh [MODE]

# ================= 环境变量优化 =================
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=0
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
    echo " train_prob - 训练概率分箱模型 (迁移学习)"
    echo " test_prob  - 测试概率分箱模型"
    echo " infer_prob - 使用概率分箱模型进行推理"
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
            --batch_size 4 \
            --accumulate_grad_batches 8 \
            --num_workers 8 \
            \
            --in_shape 10 54 256 256 \
            --aft_seq_length 20 \
            --max_epochs 40 \
            --opt adamw \
            --lr 8e-4 \
            --sched cosine \
            --min_lr 1e-5 \
            --warmup_epoch 2 \
            \
            --model_type mamba \
            --hid_S 128 \
            --hid_T 512 \
            --N_S 4 \
            --N_T 12 \
            --mlp_ratio 8.0 \
            --drop 0.0 \
            --drop_path 0.1 \
            --spatio_kernel_enc 5 \
            --spatio_kernel_dec 5 \
            --loss_weight_l1 10.0 \
            --loss_weight_csi 0.5 \
            \
            --use_curriculum_learning true \
            --early_stop_patience 15 \
            --early_stop_monitor val_score \
            --early_stop_mode max \
            --accelerator cuda \
            --devices 1,2,3\
            --precision bf16-mixed \
            --gradient_clip_val 0.5 \
            --gradient_clip_algorithm norm
        ;;
        
    # ============================================================
    # 2. 测试 SimVP 基座 - [保持原样]
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
            --accelerator cuda:0
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
    # 4. 训练概率分箱模型 (Stage 2: Transfer Learning)
    # ============================================================
    "train_prob")
        echo "--------------------------------------------------------"
        echo "🚀 开始微调 Probabilistic Mamba (Transfer Learning)..."
        echo "--------------------------------------------------------"
        
        # 注意: --base_ckpt_dir 指向 Stage 1 的输出目录，用于自动查找 best.ckpt 进行加载
        python run/train_scwds_prob.py \
            --data_path data/samples.jsonl \
            --base_ckpt_dir ./output/simvp \
            --save_dir ./output/prob_simvp \
            --in_shape 10 54 256 256 \
            --aft_seq_length 20 \
            --num_bins 64 \
            \
            --batch_size 4 \
            --num_workers 8 \
            --max_epochs 30 \
            \
            --lr 1e-4 \
            --min_lr 1e-6 \
            --warmup_epoch 2 \
            \
            --early_stop_patience 10 \
            --accelerator cuda \
            --devices 1,2,3 \
            --precision bf16-mixed
        ;;

    # ============================================================
    # 5. 测试概率分箱模型
    # ============================================================
    "test_prob")
        echo "----------------------------------------"
        echo "🧪 开始测试 Probabilistic Mamba 模型..."
        echo "----------------------------------------"
        
        python run/test_scwds_prob.py \
            --data_path data/samples.jsonl \
            --in_shape 10 54 256 256 \
            --aft_seq_length 20 \
            --num_bins 64 \
            --save_dir ./output/prob_simvp \
            --num_samples 10 \
            --accelerator cuda:0
        ;;

    # ============================================================
    # 6. 推理概率分箱模型
    # ============================================================
    "infer_prob")
        echo "----------------------------------------"
        echo "🔮 开始推理 Probabilistic Mamba 模型..."
        echo "----------------------------------------"
        
        python run/infer_scwds_prob.py \
            --data_path data/samples.testset.jsonl \
            --in_shape 20 54 256 256 \
            --num_bins 64 \
            --save_dir ./output/prob_simvp \
            --accelerator cuda:0 \
            --vis \
            --vis_output ./output/prob_simvp/vis_infer
        ;;
        
esac

echo "✅ 操作完成！"