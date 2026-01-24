# src/utils/generate_configs.py

import os
import yaml
import copy
from itertools import product

# 基础模板
BASE_CONFIG = {
    "experiment": {
        "name": "template", 
        "device": "cuda",  # Server default
        "seed": 42
    },
    "latent_space": {
        "info_dim": 10,
        "redundancy_factor": 2,
        "ecc_type": "repetition"
    },
    "kernels": {
        "type": "ng_sm",
        "num_mixtures": 4,
        "rff_samples": 500
    },
    "views": {
        "fac": { "input_dim": 216, "likelihood": "gaussian" },
        "fou": { "input_dim": 76,  "likelihood": "gaussian" },
        "kar": { "input_dim": 64,  "likelihood": "gaussian" },
        "pix": { "input_dim": 240, "likelihood": "gaussian" },
        "zer": { "input_dim": 47,  "likelihood": "gaussian" },
        "mor": { "input_dim": 6,   "likelihood": "gaussian" }
    },
    "training": {
        "batch_size": 256,
        "lr": 0.01,
        "epochs": 500,
        "log_interval": 50 # Less output for server
    }
}

def generate_grid():
    output_dir = "configs/grid"
    os.makedirs(output_dir, exist_ok=True)
    
    
    # 实验变量 (Grid Search)
    redundancy_factors = [2, 5] # Exclude 1 for now, handle separately? Or just handle L=1 as Uncoded
    rff_samples = [500] # Simplification for demo, or keep [500, 2000, 5000]
    info_dims = [10]
    
    # Dual-Path factors
    ecc_types = ['repetition', 'random_gaussian']
    inference_modes = ['direct', 'amortized']
    
    # 1. Uncoded Baseline (L=1, ECC=none, Direct)
    # We can mix uncoded with amortized too!
    
    # Let's generate a focused list:
    # Path A: Direct + Random Gaussian (vs Repetition)
    # Path B: Amortized + Repetition (vs Direct)
    
    configs = []
    
    # A. Uncoded Baselines
    for inf_mode in inference_modes:
        configs.append({
            'L': 1, 'type': 'none', 'inf': inf_mode, 'name': f'Uncoded_{inf_mode}'
        })

    # B. Coded Configs
    combinations = list(product(redundancy_factors, ecc_types, inference_modes))
    for r, ecc, inf in combinations:
        configs.append({
            'L': r, 'type': ecc, 'inf': inf, 
            'name': f'Coded_L{r}_{ecc}_{inf}'
        })

    print(f"Generating {len(configs)} configurations...")
    
    for c in configs:
        cfg = copy.deepcopy(BASE_CONFIG)
        
        # Set Params
        cfg['latent_space']['redundancy_factor'] = c['L']
        cfg['latent_space']['ecc_type'] = c['type']
        # cfg['training']['inference_mode'] = c['inf'] # Wait, model init param, need to pass via config?
        # The MODEL accepts init param. We need to put it in config YAML so Trainer can read it.
        # Let's assume Trainer reads cfg['model']['inference_mode'] or similar.
        # We need to add 'inference_mode' to config structure.
        if 'model' not in cfg: cfg['model'] = {}
        cfg['model']['inference_mode'] = c['inf']
        
        # Name
        exp_name = f"cng_{c['name']}"
        cfg['experiment']['name'] = exp_name
        
        # Save
        filename = f"{output_dir}/{exp_name}.yaml"
        with open(filename, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
            
        print(f" -> Created {filename}")

if __name__ == "__main__":
    generate_grid()
