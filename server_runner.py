# server_runner.py

import os
import glob
import time
import subprocess
import sys
import multiprocessing
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

PYTHON_EXEC = sys.executable 

def get_gpu_count():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except ImportError:
        try:
            output = subprocess.check_output(['nvidia-smi', '-L']).decode('utf-8')
            return len([line for line in output.split('\n') if line.strip()])
        except:
            pass
    return 0

def get_pending_experiments(config_dir, checkpoint_dir):
    all_configs = sorted(glob.glob(os.path.join(config_dir, "*.yaml")))
    pending = []
    
    for cfg_path in all_configs:
        basename = os.path.basename(cfg_path).replace(".yaml", "")
        # Check if completed in the specific checkpoint directory
        # We assume if any folder starts with basename and has final_model.pth, it's done.
        # But wait, with timestamps, folder names vary. 
        # If we are in a strict experiment isolation, the checkpoint_dir is specific to this run.
        # So we should check if `checkpoint_dir` itself has a subdirectory for this model?
        # Actually, engine.py creates `checkpoint_dir/{exp_name}_{timestamp}`.
        
        candidates = glob.glob(os.path.join(checkpoint_dir, f"{basename}_*"))
        is_done = False
        for c in candidates:
            if os.path.exists(os.path.join(c, "final_model.pth")):
                is_done = True
                break
        
        if not is_done:
            pending.append(cfg_path)
            
    return pending

def run_experiment(cfg_path, gpu_id, checkpoint_dir, log_dir):
    basename = os.path.basename(cfg_path).replace(".yaml", "")
    print(f"[Runner] Starting: {basename} on GPU:{gpu_id if gpu_id is not None else 'CPU'}")
    
    start_time = time.time()
    
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    cmd = [PYTHON_EXEC, "-m", "src.trainer.engine", "--config", cfg_path]
    if checkpoint_dir:
        cmd.extend(["--checkpoint_dir", checkpoint_dir])
    if log_dir:
        cmd.extend(["--log_dir", log_dir])
    
    try:
        # Determine strict or console logging
        max_workers = int(os.environ.get("MAX_WORKERS", "1"))
        
        if max_workers <= 1:
            print(f"[Runner] --- Console Output for {basename} ---")
            subprocess.check_call(cmd, env=env)
        else:
            # If parallel, verify log_dir exists
            target_log_dir = log_dir if log_dir else "logs"
            os.makedirs(target_log_dir, exist_ok=True)
            log_file = os.path.join(target_log_dir, f"run_{basename}.log")
            with open(log_file, "w") as f:
                subprocess.check_call(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
            
        duration = time.time() - start_time
        print(f"[Runner] ✅ COMPLETED: {basename} in {duration:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Runner] ❌ FAILED: {basename}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="configs/caltech_batch", help="Directory containing config yamls")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to store checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to store logs")
    parser.add_argument("--max_workers", type=int, default=None, help="Number of parallel workers")
    args = parser.parse_args()

    if not os.path.exists(args.config_dir):
        print(f"Config dir {args.config_dir} not found.")
        return

    # Ensure output dirs exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    pending = get_pending_experiments(args.config_dir, args.checkpoint_dir)
    num_pending = len(pending)
    print(f"\nFound {num_pending} pending experiments in {args.config_dir}")
    print(f"Target Checkpoints: {args.checkpoint_dir}")
    
    if num_pending == 0:
        return

    num_gpus = get_gpu_count()
    if args.max_workers is not None:
        max_workers = args.max_workers
    else:
        max_workers = int(os.environ.get("MAX_WORKERS", num_gpus if num_gpus > 0 else 1))
    
    print(f"Detected {num_gpus} GPUs. Running with {max_workers} parallel workers.\n")

    if max_workers <= 1:
        for i, cfg in enumerate(pending):
            print(f"--- Progress: {i+1}/{num_pending} ---")
            run_experiment(cfg, 0 if num_gpus > 0 else None, args.checkpoint_dir, args.log_dir)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_cfg = {}
            for i, cfg in enumerate(pending):
                gpu_id = i % num_gpus if num_gpus > 0 else None
                future = executor.submit(run_experiment, cfg, gpu_id, args.checkpoint_dir, args.log_dir)
                future_to_cfg[future] = cfg
            
            completed = 0
            for future in as_completed(future_to_cfg):
                completed += 1
                cfg = future_to_cfg[future]
                print(f"[Runner] Global Progress: {completed}/{num_pending}")

    print("\nAll experiments finished.")

if __name__ == "__main__":
    main()
