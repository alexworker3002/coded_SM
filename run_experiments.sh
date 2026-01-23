#!/bin/bash

# 激活环境 (针对 shell 脚本)
# 注意: 在 run_command 中我们通常直接使用完整 python 路径，但在脚本中我们可以假设环境已激活或指定 python
PYTHON_EXEC="/Users/ice/miniforge3/envs/cng_mvlvm/bin/python"

echo "========================================================"
echo "Starting Experiment 1: Coded CNG-MV-GPLVM (Repetition)"
echo "========================================================"
$PYTHON_EXEC -m src.trainer.engine --config configs/mfeat_default.yaml

echo ""
echo "========================================================"
echo "Starting Experiment 2: Uncoded Baseline (Identity)"
echo "========================================================"
$PYTHON_EXEC -m src.trainer.engine --config configs/mfeat_uncoded.yaml

echo ""
echo "✅ All experiments completed."
