import os
import yaml
import copy

BASE_CONFIG = {
    "experiment": {
        "name": "template", 
        "device": "auto",  
        "seed": 42
    },
    "latent_space": {
        "info_dim": 2, # Will be overwritten
        "redundancy_factor": 2, # Will be overwritten
        "ecc_type": "repetition"
    },
    "model": {
        "inference_mode": "amortized",
        "encoder_type": "mlp"
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
        "epochs": 1000, 
        "log_interval": 50
    }
}

def generate_dim_batch():
    output_dir = "configs/dim_batch"
    os.makedirs(output_dir, exist_ok=True)
    
    dims = [2, 5, 10]
    configs = []
    
    for z in dims:
        # 1. SMLVM (Uncoded, L=1, Direct)
        configs.append({
            'name': f'smlvm_Z{z}_L1',
            'z': z, 'L': 1, 'ecc': 'none', 'inf': 'direct', 'enc': 'mlp' 
        })
        
        # 2. VAE-MLP (L=2, Repetition)
        configs.append({
            'name': f'vae_mlp_Z{z}_L2_rep',
            'z': z, 'L': 2, 'ecc': 'repetition', 'inf': 'amortized', 'enc': 'mlp'
        })
        
        # 3. VAE-MLP (L=2, Random Gaussian)
        configs.append({
            'name': f'vae_mlp_Z{z}_L2_random',
            'z': z, 'L': 2, 'ecc': 'random_gaussian', 'inf': 'amortized', 'enc': 'mlp'
        })
        
    print(f"Generating {len(configs)} configurations in {output_dir}...")
    
    for c in configs:
        cfg = copy.deepcopy(BASE_CONFIG)
        
        # Params
        cfg['latent_space']['info_dim'] = c['z'] # Set Info Dim
        cfg['latent_space']['redundancy_factor'] = c['L']
        cfg['latent_space']['ecc_type'] = c['ecc']
        cfg['model']['inference_mode'] = c['inf']
        cfg['model']['encoder_type'] = c['enc']
        
        # Name
        exp_name = c['name']
        cfg['experiment']['name'] = exp_name
        
        # Save
        filename = f"{output_dir}/{exp_name}.yaml"
        with open(filename, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
            
        print(f" -> Created {filename}")
        
    # Generate Debug Config (Z=2)
    debug_cfg = copy.deepcopy(BASE_CONFIG)
    debug_cfg['experiment']['name'] = 'debug_vae_mlp_Z2'
    debug_cfg['latent_space']['info_dim'] = 2
    debug_cfg['training']['epochs'] = 1
    debug_cfg['latent_space']['ecc_type'] = 'random_gaussian'
    debug_filename = f"{output_dir}/debug_vae_mlp_Z2.yaml"
    with open(debug_filename, 'w') as f:
        yaml.dump(debug_cfg, f, default_flow_style=False)
    print(f" -> Created Debug: {debug_filename}")

if __name__ == "__main__":
    generate_dim_batch()
