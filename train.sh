#!/bin/bash

# EEG分类训练脚本（集成向量数据库）

# 设置实验名称
EXP_NAME="eeg_classification_with_vector_db"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_DIR="./experiments/${EXP_NAME}_${TIMESTAMP}"

# 创建实验目录
mkdir -p $EXP_DIR

echo "========================================"
echo "EEG分类实验（集成向量数据库）"
echo "实验目录: $EXP_DIR"
echo "开始时间: $(date)"
echo "========================================"

# 使用配置文件运行
python main.py \
  --config ./config.yaml \
  --save_dir $EXP_DIR \
  --subject 0 \
  --model Interpretable_mamba \
  --mamba_dim 128 \
  --batch_size 64 \
  --dropout 0.3 \
  --epochs 2000 \
  --lr 1e-3 \
  --weight_decay 0.01 \
  --mixup \
  --mixup_alpha 0.4 \
  --grad_clip 1.0 \
  --patience 500 \
  --augment \
  --augment_factor 3 \
  --scheduler cosine \
  --label_smoothing 0.15 \
  --fre_filter \
  --optimizer adamw \
  --enable_vector_db \
  --db_type faiss \
  --feature_dim 128 \
  --retrieval_topk 5 \
  --use_db_for_few_shot \
  --save_model \
  --plot_curves \
  --plot_cm \
  --plot_retrieval \
  --verbose 1

# 复制配置文件到实验目录
cp ./config.yaml $EXP_DIR/

echo "========================================"
echo "实验完成!"
echo "结束时间: $(date)"
echo "结果保存在: $EXP_DIR"
echo "========================================"

# 生成实验报告
if [ -f "$EXP_DIR/results.json" ]; then
    echo "生成实验报告..."
    python -c "
import json
import os
exp_dir = '$EXP_DIR'
with open(os.path.join(exp_dir, 'results.json'), 'r') as f:
    results = json.load(f)
print('\\n实验摘要:')
print('='*40)
print(f'测试准确率: {results[\"test_acc\"]:.4f}')
print(f'测试Kappa: {results[\"test_kappa\"]:.4f}')
print(f'测试F1: {results[\"test_f1\"]:.4f}')
print(f'最佳验证准确率: {results[\"best_val_acc\"]:.4f}')
print('='*40)
"
fi
