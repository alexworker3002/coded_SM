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
    
    # Check if config is a symlink pointing to an experiment directory structure
    experiment_dir = None
    if os.path.islink(cfg_path):
        real_path = os.path.realpath(cfg_path)
        # Assuming structure: experiment_dir/config.yaml
        possible_exp_dir = os.path.dirname(real_path)
        if os.path.exists(os.path.join(possible_exp_dir, "run.sh")):
            experiment_dir = possible_exp_dir
    
    if experiment_dir:
        print(f"[Runner] Detected Independent Experiment Dir: {experiment_dir}")
        cmd.extend(["--experiment_dir", experiment_dir])
        # We generally do NOT pass checkpoint_dir/log_dir if experiment_dir is set,
        # unless we want to override. engine.py prioritizes experiment_dir internal paths.
    else:
        if checkpoint_dir:
            cmd.extend(["--checkpoint_dir", checkpoint_dir])
        if log_dir:
            cmd.extend(["--log_dir", log_dir])
    
    try:
        # Always output to console for real-time progress as requested
        # Even in parallel mode, this will show interleaved progress bars
        subprocess.check_call(cmd, env=env)
            
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
    parser.add_argument("--max_workers", type=int, default=None, help="Number of parallel workers (auto if None)")
    parser.add_argument("--auto_analyze", action="store_true", help="Automatically run analysis after training")
    parser.add_argument("--result_name", type=str, default=None, help="Custom name for result folder (auto-generated if None)")
    args = parser.parse_args()


    if not os.path.exists(args.config_dir):
        print(f"Config dir {args.config_dir} not found.")
        return

    import datetime
    import shutil
    timestamp = datetime.datetime.now().strftime('%Y%b%d_%H%M%S')
    
    # Auto-detect batch name from config_dir (e.g., configs/mfeat_z20 -> mfeat_z20)
    batch_name_base = args.result_name if args.result_name else os.path.basename(args.config_dir.rstrip('/'))
    batch_id = f"{batch_name_base}_{timestamp}"
    
    # Create batch-specific subdirectories for consistent structure
    batch_checkpoint_dir = os.path.join(args.checkpoint_dir, batch_id)
    batch_log_dir = os.path.join(args.log_dir, batch_id)
    batch_config_archive = os.path.join("configs/archive", batch_id)
    
    os.makedirs(batch_checkpoint_dir, exist_ok=True)
    os.makedirs(batch_log_dir, exist_ok=True)
    os.makedirs(batch_config_archive, exist_ok=True)
    
    # Archive configs for reproducibility
    source_configs = glob.glob(os.path.join(args.config_dir, "*.yaml"))
    for cfg in source_configs:
        shutil.copy(cfg, batch_config_archive)
    
    print(f"Batch ID: {batch_id}")
    print(f"Checkpoints: {batch_checkpoint_dir}")
    print(f"Logs: {batch_log_dir}")
    print(f"Config Archive: {batch_config_archive}")

    pending = get_pending_experiments(args.config_dir, batch_checkpoint_dir)
    num_pending = len(pending)
    print(f"\nFound {num_pending} pending experiments in {args.config_dir}")
    
    if num_pending == 0:
        return

    num_gpus = get_gpu_count()
    if args.max_workers is not None:
        max_workers = args.max_workers
    else:
        # Auto-adjust: find largest divisor of num_pending that's <= num_gpus
        # This ensures even distribution (e.g., 10 tasks with 1 GPU → 5 workers)
        max_workers_env = int(os.environ.get("MAX_WORKERS", num_gpus if num_gpus > 0 else 1))
        max_workers = 1
        for w in range(min(num_pending, max_workers_env), 0, -1):
            if num_pending % w == 0:
                max_workers = w
                break
    
    batches_count = (num_pending + max_workers - 1) // max_workers
    print(f"Detected {num_gpus} GPUs. Running {num_pending} tasks with {max_workers} workers ({batches_count} batches).\n")

    if max_workers <= 1:
        for i, cfg in enumerate(pending):
            print(f"--- Progress: {i+1}/{num_pending} ---")
            run_experiment(cfg, 0 if num_gpus > 0 else None, batch_checkpoint_dir, batch_log_dir)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_cfg = {}
            for i, cfg in enumerate(pending):
                gpu_id = i % num_gpus if num_gpus > 0 else None
                future = executor.submit(run_experiment, cfg, gpu_id, batch_checkpoint_dir, batch_log_dir)
                future_to_cfg[future] = cfg
            
            completed = 0
            for future in as_completed(future_to_cfg):
                completed += 1
                cfg = future_to_cfg[future]
                print(f"[Runner] Global Progress: {completed}/{num_pending}")


    print("\nAll experiments finished.")
    
    # Auto-analysis if requested
    if args.auto_analyze:
        print("\n" + "="*60)
        print("[Runner] Starting automated analysis...")
        print("="*60)
        
        result_folder = os.path.join("results", batch_id)
        os.makedirs(result_folder, exist_ok=True)
        
        # Run analysis - point to the specific batch checkpoints
        analysis_cmd = [
            PYTHON_EXEC, "auto_analyze.py",
            "--experiments", os.path.join(batch_checkpoint_dir, "*"),
            "--output", result_folder
        ]
        
        try:
            subprocess.check_call(analysis_cmd)
            print(f"\n✅ Analysis complete. Results in: {result_folder}")
        except subprocess.CalledProcessError:
            print(f"\n❌ Analysis failed. Check logs.")

if __name__ == "__main__":
    main()
