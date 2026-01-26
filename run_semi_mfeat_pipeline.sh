#!/bin/bash
set -e

# Isolated config and results folders
export CONFIG_DIR="configs/semi_mfeat"
RESULTS_DIR="results/semi_mfeat"

echo ">>> [1/3] Generating Semi-Amortized Mfeat Configurations (Z=10,20 | L=2,5,10)..."
python src/utils/generate_semi_mfeat.py

echo ">>> [2/3] Training Models (Sequential Real-time Output)..."
mkdir -p logs/mfeat

# Set to 1 for real-time console reporting
export MAX_WORKERS=1 
python server_runner.py 

echo "Training Complete."

echo ">>> [3/3] Running Analysis..."
python -m src.analysis.compare_semi_mfeat
echo ">>> Pipeline Finished. Results in $RESULTS_DIR"
