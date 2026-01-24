# server_runner.py

import os
import glob
import time
import subprocess
import sys

# Configuration
CONFIG_DIR = "configs/caltech_batch"
CHECKPOINT_DIR = "checkpoints"
PYTHON_EXEC = sys.executable # Use current python

def get_pending_experiments():
    all_configs = sorted(glob.glob(os.path.join(CONFIG_DIR, "*.yaml")))
    pending = []
    
    for cfg_path in all_configs:
        # Extract exp name from filename
        basename = os.path.basename(cfg_path).replace(".yaml", "")
        
        # Check if already done (look for final_model.pth in any folder starting with exp_name)
        # Checkpoints format: checkpoints/exp_name_TIMESTAMP
        candidates = glob.glob(os.path.join(CHECKPOINT_DIR, f"{basename}_*"))
        
        is_done = False
        for c in candidates:
            if os.path.exists(os.path.join(c, "final_model.pth")):
                is_done = True
                break
        
        if not is_done:
            pending.append(cfg_path)
        else:
            print(f"Skipping {basename} (Already completed).")
            
    return pending

def run_experiment(cfg_path):
    print(f"\n>>> Starting Experiment: {cfg_path}")
    start_time = time.time()
    
    # Run Engine
    # Ensure module path is correct
    cmd = [PYTHON_EXEC, "-m", "src.trainer.engine", "--config", cfg_path]
    
    try:
        # Stream output to console
        subprocess.check_call(cmd)
        duration = time.time() - start_time
        print(f">>> Completed in {duration:.2f} seconds.")
    except subprocess.CalledProcessError as e:
        print(f">>> FAILED: {cfg_path} with error code {e.returncode}")

def main():
    if not os.path.exists(CONFIG_DIR):
        print(f"Config dir {CONFIG_DIR} not found. Did you run 'src/utils/generate_configs.py'?")
        return

    pending = get_pending_experiments()
    print(f"\nFound {len(pending)} pending experiments.")
    
    for i, cfg in enumerate(pending):
        print(f"\n--- Progress: {i+1}/{len(pending)} ---")
        run_experiment(cfg)
        
    print("\nAll experiments finished.")

if __name__ == "__main__":
    main()
