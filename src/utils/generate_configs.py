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
    # Control Variables
    redundancy_factors = [1, 2, 5]
    rff_samples = [500, 2000, 5000]
    info_dims = [10, 20]
    
    # Generate Combinations
    combinations = list(product(redundancy_factors, rff_samples, info_dims))
    
    print(f"Generating {len(combinations)} configurations...")
    
    for r, s, d in combinations:
        cfg = copy.deepcopy(BASE_CONFIG)
        
        # Determine Type
        if r == 1:
            exp_type = "Uncoded"
            cfg['latent_space']['ecc_type'] = "none"
        else:
            exp_type = f"Coded_L{r}"
            cfg['latent_space']['ecc_type'] = "repetition"
            
        # Set Values
        cfg['latent_space']['redundancy_factor'] = r
        cfg['latent_space']['info_dim'] = d
        cfg['kernels']['rff_samples'] = s
        
        # Name: type_dim_samples
        exp_name = f"cng_feat_{exp_type}_D{d}_S{s}"
        cfg['experiment']['name'] = exp_name
        
        # Save
        filename = f"{output_dir}/{exp_name}.yaml"
        with open(filename, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
            
        print(f" -> Created {filename}")

if __name__ == "__main__":
    generate_grid()
