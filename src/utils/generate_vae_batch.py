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
        "info_dim": 2, # Fixed to 2 for this batch
        "redundancy_factor": 5,
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
        "epochs": 1000, # Longer training for VAE?
        "log_interval": 50
    }
}

def generate_batch():
    output_dir = "configs/vae_batch"
    os.makedirs(output_dir, exist_ok=True)
    
    configs = []
    
    # 1. VAE-CNN (L=5, Random, Z=2)
    configs.append({
        'name': 'vae_cnn_L5_random',
        'L': 5, 'ecc': 'random_gaussian', 'inf': 'amortized', 'enc': 'cnn'
    })
    
    # 2. VAE-CNN (L=5, Repetition, Z=2)
    configs.append({
        'name': 'vae_cnn_L5_rep',
        'L': 5, 'ecc': 'repetition', 'inf': 'amortized', 'enc': 'cnn'
    })
    
    # 3. VAE-MLP (L=5, Random, Z=2)
    configs.append({
        'name': 'vae_mlp_L5_random',
        'L': 5, 'ecc': 'random_gaussian', 'inf': 'amortized', 'enc': 'mlp'
    })
    
    # 4. VAE-MLP (L=5, Repetition, Z=2)
    configs.append({
        'name': 'vae_mlp_L5_rep',
        'L': 5, 'ecc': 'repetition', 'inf': 'amortized', 'enc': 'mlp'
    })
    
    # 5. SMLVM (L=1, Uncoded, Z=2, Direct) - Yang et al. Reference
    configs.append({
        'name': 'smlvm_uncoded_L1',
        'L': 1, 'ecc': 'none', 'inf': 'direct', 'enc': 'mlp' # Enc ignored for direct
    })
    
    print(f"Generating {len(configs)} configurations in {output_dir}...")
    
    for c in configs:
        cfg = copy.deepcopy(BASE_CONFIG)
        
        # Params
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
        
    # Generate Debug Config
    debug_cfg = copy.deepcopy(BASE_CONFIG)
    debug_cfg['experiment']['name'] = 'debug_vae_cnn'
    debug_cfg['training']['epochs'] = 1
    debug_cfg['latent_space']['ecc_type'] = 'random_gaussian'
    debug_cfg['model']['encoder_type'] = 'cnn'
    debug_filename = f"{output_dir}/debug_vae_cnn.yaml"
    with open(debug_filename, 'w') as f:
        yaml.dump(debug_cfg, f, default_flow_style=False)
    print(f" -> Created Debug: {debug_filename}")

if __name__ == "__main__":
    generate_batch()
