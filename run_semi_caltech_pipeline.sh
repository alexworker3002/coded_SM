#!/bin/bash
set -e

# Isolated config and results folders
export CONFIG_DIR="configs/semi_caltech"
RESULTS_DIR="results/semi_caltech"

echo ">>> [1/3] Generating Semi-Amortized Caltech Configurations (Z=20,40 | L=2,5)..."
python src/utils/generate_semi_caltech.py

echo ">>> [2/3] Training Models (Sequential Real-time Output)..."
mkdir -p logs/semi_caltech

# Set to 1 for real-time console reporting as requested previously
export MAX_WORKERS=1 
python server_runner.py 

echo "Training Complete."

echo ">>> [3/3] Running Analysis..."
python -m src.analysis.compare_semi_caltech
echo ">>> Pipeline Finished. Results in $RESULTS_DIR"
