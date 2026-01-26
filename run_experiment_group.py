#!/usr/bin/env python
import os
import sys
import argparse
import datetime
import subprocess

# Local imports (using subprocess for robust env handling)
EXEC_PYTHON = sys.executable

def run_cmd(cmd_list, env=None, check=True):
    print(f"\n[GroupRunner] Executing: {' '.join(cmd_list)}")
    if env is None:
        env = os.environ.copy()
    subprocess.run(cmd_list, env=env, check=check)

def main():
    parser = argparse.ArgumentParser(description="Run an isolated experiment group.")
    parser.add_argument("--tag", type=str, required=True, help="Tag for this experiment run (e.g., 'Benchmark_v1')")
    parser.add_argument("--datasets", nargs="+", default=["mfeat", "caltech", "100leaves"], help="Datasets to run")
    parser.add_argument("--dry_run", action="store_true", help="Generate configs but do not train")
    parser.add_argument("--skip_train", action="store_true", help="Skip training, only run analysis")
    args = parser.parse_args()

    # 1. Setup Experiment Directory
    timestamp = datetime.datetime.now().strftime('%Y%b%d_%H-%M-%S')
    exp_dir_name = f"EXP_{timestamp}_{args.tag}"
    exp_root = os.path.abspath(os.path.join("experiments", exp_dir_name))
    
    config_root = os.path.join(exp_root, "configs")
    ckpt_root = os.path.join(exp_root, "checkpoints")
    log_root = os.path.join(exp_root, "logs")
    result_root = os.path.join(exp_root, "results")
    
    # Ensure directories exist
    os.makedirs(exp_root, exist_ok=True)
    os.makedirs(config_root, exist_ok=True)
    os.makedirs(ckpt_root, exist_ok=True)
    os.makedirs(log_root, exist_ok=True)
    os.makedirs(result_root, exist_ok=True)
    
    print(f"===========================================================")
    print(f"🚀  STARTING EXPERIMENT GROUP: {exp_dir_name}")
    print(f"📁  Root Path: {exp_root}")
    print(f"===========================================================")

    # Define Dataset Mappings
    dataset_map = {
        "mfeat": {
            "gen_script": "src/utils/generate_semi_mfeat.py",
            "comp_script": "src.analysis.compare_semi_mfeat",
            "config_sub": "mfeat"
        },
        "caltech": {
            "gen_script": "src/utils/generate_semi_caltech.py",
            "comp_script": "src.analysis.compare_semi_caltech",
            "config_sub": "caltech101-7"
        },
        "100leaves": {
            "gen_script": "src/utils/generate_semi_100leaves.py",
            "comp_script": "src.analysis.compare_semi_100leaves",
            "config_sub": "100leaves"
        }
    }

    # 2. Generate Configs
    print("\n>>> [1/3] Generating Configurations...")
    for ds in args.datasets:
        if ds not in dataset_map:
            print(f"Warning: Unknown dataset {ds}, skipping.")
            continue
            
        meta = dataset_map[ds]
        target_conf_dir = os.path.join(config_root, meta['config_sub'])
        
        cmd = [EXEC_PYTHON, meta['gen_script'], "--output_dir", target_conf_dir]
        run_cmd(cmd)

    if args.dry_run:
        print("\n[Dry Run] Configs generated. Exiting.")
        return

    # 3. Training
    if not args.skip_train:
        print("\n>>> [2/3] Running Training (Sequential)...")
        # We run server_runner for EACH dataset config folder to ensure controlled execution
        # server_runner takes --config_dir, --checkpoint_dir, --log_dir
        
        # NOTE: server_runner.py is in root
        runner_script = "server_runner.py"
        
        # Enforce correct environment via wrapper or subprocess?
        # Assuming we are running this script in the correct environment (cng_mvlvm_server).
        # We delegate to server_runner.
        
        for ds in args.datasets:
            if ds not in dataset_map: continue
            meta = dataset_map[ds]
            target_conf_dir = os.path.join(config_root, meta['config_sub'])
            # Checkpoints will be stored in ckpt_root explicitly (engine appends dataset? No, engine appends exp_name/time)
            # Actually engine.py with override uses ckpt_root/{exp_name}_{time} directly.
            # So models for all datasets will be mixed in ckpt_root?
            # Better to separate them: ckpt_root/dataset/
            
            ds_ckpt_dir = os.path.join(ckpt_root, meta['config_sub'])
            ds_log_dir = os.path.join(log_root, meta['config_sub'])
            
            os.makedirs(ds_ckpt_dir, exist_ok=True)
            os.makedirs(ds_log_dir, exist_ok=True)

            print(f"\n---> Training dataset: {ds}")
            
            # Use MAX_WORKERS=1 for stability
            env = os.environ.copy()
            env['MAX_WORKERS'] = '1'
            
            cmd = [
                EXEC_PYTHON, runner_script,
                "--config_dir", target_conf_dir,
                "--checkpoint_dir", ds_ckpt_dir,
                "--log_dir", ds_log_dir,
                "--max_workers", "1"
            ]
            run_cmd(cmd, env=env)
    
    # 4. Analysis
    print("\n>>> [3/3] Running Analysis...")
    for ds in args.datasets:
        if ds not in dataset_map: continue
        meta = dataset_map[ds]
        
        ds_ckpt_dir = os.path.join(ckpt_root, meta['config_sub'])
        ds_res_dir = os.path.join(result_root, meta['config_sub'])
        
        print(f"\n---> Analyzing dataset: {ds}")
        
        # comp_script is a module path
        cmd = [
            EXEC_PYTHON, "-m", meta['comp_script'],
            "--checkpoint_dir", ds_ckpt_dir,
            "--results_dir", ds_res_dir
        ]
        
        # Analysis scripts might fail if training failed, so check=False
        try:
            run_cmd(cmd, check=True)
        except subprocess.CalledProcessError:
             print(f"❌ Analysis failed for {ds}")

    print("\n" + "="*50)
    print(f"✅ EXPERIMENT GROUP COMPLETED")
    print(f"📁 Results saved to: {result_root}")
    print("="*50)

if __name__ == "__main__":
    main()
