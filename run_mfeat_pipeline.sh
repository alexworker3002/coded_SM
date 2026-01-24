#!/bin/bash
set -e

echo ">>> [1/3] Generating Mfeat Configurations (Z=10, 15, 20, 30)..."
python src/utils/generate_mfeat_batch.py

echo ">>> [2/3] Training Models (Output: logs/mfeat_pipeline.log)..."
# Override CONFIG_DIR env var for server_runner
export CONFIG_DIR="configs/mfeat_batch"
nohup python server_runner.py > logs/mfeat_pipeline.log 2>&1 &
PID=$!
echo "Training started in background (PID: $PID)."
echo "You can check progress with: tail -f logs/mfeat_pipeline.log"
echo "Wait for training to complete before running analysis."
echo "Waiting..."
wait $PID
echo "Training Complete."

echo ">>> [3/3] Running Analysis..."
python -m src.analysis.compare_for_mfeat
echo ">>> All Done! Check mfeat_benchmark_*.png and csv."
