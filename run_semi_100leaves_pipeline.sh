#!/bin/bash
set -e

# Isolated config and results folders
export CONFIG_DIR="configs/semi_100leaves"
RESULTS_DIR="results/semi_100leaves"

echo ">>> [1/3] Generating Semi-Amortized 100Leaves Configurations (Z=32,64 | L=2,5)..."
python src/utils/generate_semi_100leaves.py

echo ">>> [2/3] Training Models (Sequential Real-time Output)..."
mkdir -p logs/semi_100leaves

# Set to 1 for real-time console reporting
export MAX_WORKERS=1 
python server_runner.py 

echo "Training Complete."

echo ">>> [3/3] Running Analysis..."
python -m src.analysis.compare_semi_100leaves
echo ">>> Pipeline Finished. Results in $RESULTS_DIR"
