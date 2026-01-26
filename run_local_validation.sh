#!/bin/bash

# 一键运行本地验证 (Yang 2025 vs Semi-Amortized)
# 针对 100Leaves 数据集, Z=32, L=5

echo ">>> 1. Generating Configs..."
python src/utils/generate_semi_100leaves.py

echo ">>> 2. Training Models (Sequential)..."

# 1. Yang-Direct (Baseline)
echo "--------------------------------------------------------"
echo "Running Yang-Direct (Z=32, L=5)..."
python src/trainer/engine.py --config configs/semi_100leaves/yang_direct_Z32_L5.yaml

# 2. Yang-Amortized-MLP
echo "--------------------------------------------------------"
echo "Running Yang-Amortized-MLP (Z=32, L=5)..."
python src/trainer/engine.py --config configs/semi_100leaves/yang_amortized_mlp_Z32_L5.yaml

# 3. Semi-MLP-Rep
echo "--------------------------------------------------------"
echo "Running Semi-MLP-Rep (Z=32, L=5)..."
python src/trainer/engine.py --config configs/semi_100leaves/semi_mlp_Z32_L5_rep.yaml

# 4. Semi-MLP-Rand
echo "--------------------------------------------------------"
echo "Running Semi-MLP-Rand (Z=32, L=5)..."
python src/trainer/engine.py --config configs/semi_100leaves/semi_mlp_Z32_L5_random.yaml

echo ">>> 3. Evaluating Results..."
# 使用简化版 Evaluation 脚本（避免 Mac OpenMP 冲突）
OMP_NUM_THREADS=1 python quick_eval.py

echo ">>> DONE! Check results above."
