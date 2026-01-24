#!/bin/bash
set -e

# Isolated config and results folders
export CONFIG_DIR="configs/caltech_v2_batch"
RESULTS_DIR="results/caltech_v2"

echo ">>> [1/3] Generating Caltech v2 Configurations (Z=10, 20, 40, 80)..."
python src/utils/generate_caltech_v2_batch.py

echo ">>> [2/3] Training Models (Multi-GPU Parallel)..."
# Ensure logs dir exists
mkdir -p logs/caltech_v2

# Run server_runner with CONFIG_DIR env var
# It will automatically detect GPUs and run in parallel
nohup python server_runner.py > logs/caltech_v2/pipeline.log 2>&1 &
PID=$!

echo "Training started in background (PID: $PID)."
echo "Log: logs/caltech_v2/pipeline.log"
echo "Waiting for completion..."
wait $PID
echo "Training Complete."

echo ">>> [3/3] Running Analysis..."
python -m src.analysis.compare_for_caltech_v2
echo ">>> Pipeline Finished. Results in $RESULTS_DIR"
