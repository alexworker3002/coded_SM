# server_runner.py

import os
import glob
import time
import subprocess
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration
CONFIG_DIR = os.environ.get("CONFIG_DIR", "configs/caltech_batch")
CHECKPOINT_DIR = "checkpoints"
PYTHON_EXEC = sys.executable # Use current python

def get_gpu_count():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except ImportError:
        try:
            # Fallback to nvidia-smi if torch not installed in main env
            output = subprocess.check_output(['nvidia-smi', '-L']).decode('utf-8')
            return len([line for line in output.split('\n') if line.strip()])
        except:
            pass
    return 0

def get_pending_experiments():
    all_configs = sorted(glob.glob(os.path.join(CONFIG_DIR, "*.yaml")))
    pending = []
    
    for cfg_path in all_configs:
        basename = os.path.basename(cfg_path).replace(".yaml", "")
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

def run_experiment(cfg_path, gpu_id=None):
    basename = os.path.basename(cfg_path).replace(".yaml", "")
    print(f"[Runner] Starting: {basename} on GPU:{gpu_id if gpu_id is not None else 'CPU'}")
    
    start_time = time.time()
    
    # Environment variables for the subprocess
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    cmd = [PYTHON_EXEC, "-m", "src.trainer.engine", "--config", cfg_path]
    
    try:
        # We capture output if running in parallel to avoid garbled console, 
        # or just let it print if sequential. 
        # For parallel, it's better to log to a file per experiment.
        log_file = os.path.join("logs", f"run_{basename}.log")
        os.makedirs("logs", exist_ok=True)
        
        with open(log_file, "w") as f:
            subprocess.check_call(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
            
        duration = time.time() - start_time
        print(f"[Runner] ✅ COMPLETED: {basename} in {duration:.2f}s (Log: {log_file})")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Runner] ❌ FAILED: {basename} (Check: {log_file})")
        return False

def main():
    if not os.path.exists(CONFIG_DIR):
        print(f"Config dir {CONFIG_DIR} not found.")
        return

    pending = get_pending_experiments()
    num_pending = len(pending)
    print(f"\nFound {num_pending} pending experiments.")
    
    if num_pending == 0:
        return

    num_gpus = get_gpu_count()
    # Number of parallel workers
    max_workers = int(os.environ.get("MAX_WORKERS", num_gpus if num_gpus > 0 else 1))
    
    print(f"Detected {num_gpus} GPUs. Running with {max_workers} parallel workers.\n")

    if max_workers <= 1:
        # Sequential execution
        for i, cfg in enumerate(pending):
            print(f"--- Progress: {i+1}/{num_pending} ---")
            run_experiment(cfg, gpu_id=0 if num_gpus > 0 else None)
    else:
        # Parallel execution across GPUs
        # We use a simple strategy: assign gpu_id = worker_index % num_gpus
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_cfg = {}
            for i, cfg in enumerate(pending):
                gpu_id = i % num_gpus if num_gpus > 0 else None
                future = executor.submit(run_experiment, cfg, gpu_id)
                future_to_cfg[future] = cfg
            
            completed = 0
            for future in as_completed(future_to_cfg):
                completed += 1
                cfg = future_to_cfg[future]
                print(f"[Runner] Global Progress: {completed}/{num_pending}")

    print("\nAll experiments finished.")

if __name__ == "__main__":
    main()
