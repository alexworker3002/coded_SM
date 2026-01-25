#!/bin/bash
set -e

# Isolated config and results folders
export CONFIG_DIR="configs/semi_mfeat"
RESULTS_DIR="results/semi_mfeat"

echo ">>> [1/3] Generating Semi-Amortized Configurations (Z=10)..."
python src/utils/generate_semi_mfeat.py

echo ">>> [2/3] Training Models (Multi-GPU Parallel)..."
mkdir -p logs/semi_mfeat

# Run server_runner with CONFIG_DIR env var
nohup python server_runner.py > logs/semi_mfeat/pipeline.log 2>&1 &
PID=$!

echo "Training started in background (PID: $PID)."
echo "Log: logs/semi_mfeat/pipeline.log"
echo "Waiting for completion..."
wait $PID
echo "Training Complete."

echo ">>> [3/3] Running Analysis..."
python -m src.analysis.compare_semi_mfeat
echo ">>> Pipeline Finished. Results in $RESULTS_DIR"
