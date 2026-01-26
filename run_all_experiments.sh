#!/bin/bash
set -e

# =================================================================
# Master Experiment Script - Multi-Dataset Semi-Amortized Benchmark
# Datasets: Mfeat (Z10), Caltech (Z20), 100Leaves (Z64)
# Models: Yang-Direct (GP), VAE (MSE), Semi-Rand (GP)
# Redundancy: L=2, 5
# =================================================================

# 1. Clean and Regenerate Mfeat Configs
echo ">>> [1/3] Preparing Mfeat (Z=10, L=2,5)..."
rm -rf configs/semi_mfeat
python src/utils/generate_semi_mfeat.py

# 2. Clean and Regenerate Caltech Configs
echo ">>> [2/3] Preparing Caltech (Z=20, L=2,5)..."
rm -rf configs/semi_caltech
python src/utils/generate_semi_caltech.py

# 3. Clean and Regenerate 100Leaves Configs
echo ">>> [3/3] Preparing 100Leaves (Z=64, L=2,5)..."
rm -rf configs/semi_100leaves
python src/utils/generate_semi_100leaves.py

echo ">>> Configuration Generation Complete."
echo ">>> Starting Training Sequence (Sequential Execution)..."

# 4. Run Training
export MAX_WORKERS=1 

echo ">>> Running Training for Mfeat..."
CONFIG_DIR="configs/semi_mfeat" conda run --no-capture-output -n cng_mvlvm_server python server_runner.py

echo ">>> Running Training for Caltech..."
CONFIG_DIR="configs/semi_caltech" conda run --no-capture-output -n cng_mvlvm_server python server_runner.py

echo ">>> Running Training for 100Leaves..."
CONFIG_DIR="configs/semi_100leaves" conda run --no-capture-output -n cng_mvlvm_server python server_runner.py

echo ">>> Training Complete. Running Analysis Scripts..."

# 5. Run Analysis
conda run --no-capture-output -n cng_mvlvm_server python -m src.analysis.compare_semi_mfeat
# Note: compare_semi_caltech needs update to support Z=20 only (remove Z=40 loop if present)
conda run --no-capture-output -n cng_mvlvm_server python -m src.analysis.compare_semi_caltech 
conda run --no-capture-output -n cng_mvlvm_server python -m src.analysis.compare_semi_100leaves

echo ">>> All Experiments Completed."
