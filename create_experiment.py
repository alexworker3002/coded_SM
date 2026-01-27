
import os
import argparse
import yaml
import stat
from datetime import datetime

# Base Config Template
BASE_CONFIG = {
    "experiment": {
        "name": "template", 
        "device": "auto",  
        "seed": 42,
        "dataset": "100leaves"
    },
    "latent_space": {
        "info_dim": 32,
        "redundancy_factor": 1, 
        "ecc_type": "none"
    },
    "model": {
        "inference_mode": "direct", # direct, amortized, semi_amortized
        "encoder_type": "mlp"
    },
    "kernels": {
        "type": "ng_sm",
        "num_mixtures": 4,
        "rff_samples": 1000 
    },
    "views": {
        "shape":   { "input_dim": 64, "likelihood": "gaussian" },
        "texture": { "input_dim": 64, "likelihood": "gaussian" },
        "margin":  { "input_dim": 64, "likelihood": "gaussian" }
    },
    "training": {
        "batch_size": 128, 
        "lr": 0.01,
        "epochs": 100, 
        "log_interval": 10,
        "alignment_beta": 0.1,
        "use_gp_loss": True
    }
}

DATASET_DIMS = {
    '100leaves': {
        "shape": 64, "texture": 64, "margin": 64
    },
    'caltech101-7': {
        "WM": 40, "CENTRIST": 254, "LBP": 1180, "GIST": 512, "HOG": 198, "SIFT": 1000, "SS-LBP": 12
    },
    'mfeat': {
        "fou": 76, "fac": 216, "kar": 64, "pix": 240, "zer": 47, "mor": 6
    }
}

def create_experiment(args):
    # 1. Prepare Experiment Name and Directory
    timestamp = datetime.now().strftime('%Y%b%d_%H-%M-%S')
    exp_folder_name = f"{timestamp}_{args.name}"
    exp_dir = os.path.join("experiments", exp_folder_name)
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    
    print(f"[Create] Experiment Directory: {exp_dir}")
    
    # 2. Build Config
    cfg = BASE_CONFIG.copy()
    cfg['experiment']['name'] = args.name
    cfg['experiment']['dataset'] = args.dataset
    cfg['experiment']['seed'] = args.seed
    
    cfg['latent_space']['info_dim'] = args.z_dim
    cfg['latent_space']['redundancy_factor'] = args.redundancy
    cfg['latent_space']['ecc_type'] = args.ecc_type
    
    cfg['model']['inference_mode'] = args.inference
    cfg['model']['encoder_type'] = args.encoder
    
    cfg['training']['epochs'] = args.epochs
    cfg['training']['batch_size'] = args.batch_size
    cfg['training']['lr'] = args.lr
    cfg['training']['use_gp_loss'] = args.gp_loss
    
    # Adjust View Dims based on dataset
    if args.dataset in DATASET_DIMS:
        dims = DATASET_DIMS[args.dataset]
        cfg['views'] = {k: {"input_dim": v, "likelihood": "gaussian"} for k, v in dims.items()}
    else:
        print(f"[Warning] Unknown dataset '{args.dataset}', using default 100Leaves view structure. Please edit config manually if needed.")
        
    # Write Config
    config_path = os.path.join(exp_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"[Create] Config written to {config_path}")
    
    # 3. Create run.sh
    run_script_path = os.path.join(exp_dir, "run.sh")
    with open(run_script_path, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write("# Environment Setup\n")
        f.write("export OMP_NUM_THREADS=1\n")
        f.write("export KMP_DUPLICATE_LIB_OK=TRUE\n")
        f.write("export PYTHONPATH=$PYTHONPATH:$(pwd)\n\n")
        f.write("echo \"[Run] Starting Experiment: " + args.name + "\"\n")
        f.write(f"python src/trainer/engine.py --config {os.path.abspath(config_path)} --experiment_dir {os.path.abspath(exp_dir)}\n")
    
    # Make executable
    st = os.stat(run_script_path)
    os.chmod(run_script_path, st.st_mode | stat.S_IEXEC)
    print(f"[Create] Run script written to {run_script_path}")
    
    # 4. Create README.md
    readme_path = os.path.join(exp_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(f"# Experiment: {args.name}\n\n")
        f.write(f"**Date:** {timestamp}\n")
        f.write(f"**Description:** {args.description}\n\n")
        f.write("## Configuration Summary\n")
        f.write(f"- Dataset: {args.dataset}\n")
        f.write(f"- Model: {args.inference} (Encoder: {args.encoder})\n")
        f.write(f"- Latent: Z={args.z_dim}, L={args.redundancy}, ECC={args.ecc_type}\n")
        f.write(f"- GP Loss: {args.gp_loss}\n")
        
    print(f"[Create] README written to {readme_path}")
    print(f"\n>>> Setup Complete. To run:\n    {run_script_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a self-contained experiment folder.")
    
    # Basic info
    parser.add_argument("--name", type=str, required=True, help="Experiment name (slug)")
    parser.add_argument("--description", type=str, default="No description provided.", help="Experiment description")
    
    # Config Params
    parser.add_argument("--dataset", type=str, default="100leaves", choices=['100leaves', 'caltech101-7', 'mfeat'])
    parser.add_argument("--z_dim", type=int, default=32)
    parser.add_argument("--redundancy", type=int, default=1)
    parser.add_argument("--ecc_type", type=str, default="none", choices=['none', 'repetition', 'random_gaussian'])
    
    parser.add_argument("--inference", type=str, default="direct", choices=['direct', 'amortized', 'semi_amortized'])
    parser.add_argument("--encoder", type=str, default="mlp", choices=['mlp', 'cnn'])
    
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    
    # Toggles
    parser.add_argument("--use_amp", action="store_true", help="Enable Mixed Precision Training (May be unstable for GP)")
    parser.add_argument("--no_gp_loss", action="store_true", help="Disable GP Loss (use MSE)")
    
    args = parser.parse_args()
    args.gp_loss = not args.no_gp_loss

    # Add to config
    BASE_CONFIG['training']['use_amp'] = args.use_amp
    
    create_experiment(args)
