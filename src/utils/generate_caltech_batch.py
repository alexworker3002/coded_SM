import os
import yaml
import copy

BASE_CONFIG = {
    "experiment": {
        "name": "template", 
        "device": "auto",  
        "seed": 42,
        "dataset": "caltech101-7"
    },
    "latent_space": {
        "info_dim": 5, # Will be overwritten
        "redundancy_factor": 1, # Will be overwritten
        "ecc_type": "none"
    },
    "model": {
        "inference_mode": "amortized",
        "encoder_type": "mlp"
    },
    "kernels": {
        "type": "ng_sm",
        "num_mixtures": 4,
        "rff_samples": 1000 # Higher samples for complex data
    },
    "views": {
        # Input dimensions after PCA (handled by data_caltech.py)
        "gabor":    { "input_dim": 48,  "likelihood": "gaussian" },
        "wm":       { "input_dim": 40,  "likelihood": "gaussian" },
        "centrist": { "input_dim": 100, "likelihood": "gaussian" },
        "hog":      { "input_dim": 100, "likelihood": "gaussian" },
        "gist":     { "input_dim": 100, "likelihood": "gaussian" },
        "lbp":      { "input_dim": 100, "likelihood": "gaussian" }
    },
    "training": {
        "batch_size": 128, # Smaller batch for smaller dataset
        "lr": 0.01,
        "epochs": 800, 
        "log_interval": 50
    }
}

def generate_caltech_batch():
    output_dir = "configs/caltech_batch"
    os.makedirs(output_dir, exist_ok=True)
    
    dims = [5, 10]
    configs = []
    
    for z in dims:
        # 1. SMLVM (L=1, Direct)
        configs.append({
            'name': f'caltech_smlvm_Z{z}_L1',
            'z': z, 'L': 1, 'ecc': 'none', 'inf': 'direct', 'enc': 'mlp' 
        })
        
        # 2. VAE-MLP Uncoded (L=1)
        configs.append({
            'name': f'caltech_vae_mlp_Z{z}_L1_uncoded',
            'z': z, 'L': 1, 'ecc': 'none', 'inf': 'amortized', 'enc': 'mlp'
        })
        
        # 3. VAE-CNN Uncoded (L=1)
        configs.append({
            'name': f'caltech_vae_cnn_Z{z}_L1_uncoded',
            'z': z, 'L': 1, 'ecc': 'none', 'inf': 'amortized', 'enc': 'cnn'
        })

        # 4. VAE-MLP Repetition (L=2)
        configs.append({
            'name': f'caltech_vae_mlp_Z{z}_L2_rep',
            'z': z, 'L': 2, 'ecc': 'repetition', 'inf': 'amortized', 'enc': 'mlp'
        })

        # 5. VAE-CNN Repetition (L=2)
        configs.append({
            'name': f'caltech_vae_cnn_Z{z}_L2_rep',
            'z': z, 'L': 2, 'ecc': 'repetition', 'inf': 'amortized', 'enc': 'cnn'
        })

        # 6. VAE-MLP Random (L=5)
        configs.append({
            'name': f'caltech_vae_mlp_Z{z}_L5_random',
            'z': z, 'L': 5, 'ecc': 'random_gaussian', 'inf': 'amortized', 'enc': 'mlp'
        })

        # 7. VAE-CNN Random (L=5)
        configs.append({
            'name': f'caltech_vae_cnn_Z{z}_L5_random',
            'z': z, 'L': 5, 'ecc': 'random_gaussian', 'inf': 'amortized', 'enc': 'cnn'
        })
        
    print(f"Generating {len(configs)} configurations in {output_dir}...")
    
    for c in configs:
        cfg = copy.deepcopy(BASE_CONFIG)
        
        # Params
        cfg['latent_space']['info_dim'] = c['z']
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
        
    # Generate Debug Config (Z=5)
    debug_cfg = copy.deepcopy(BASE_CONFIG)
    debug_cfg['experiment']['name'] = 'debug_caltech_smlvm_Z5'
    debug_cfg['latent_space']['info_dim'] = 5
    debug_cfg['training']['epochs'] = 1
    debug_filename = f"{output_dir}/debug_caltech_smlvm_Z5.yaml"
    with open(debug_filename, 'w') as f:
        yaml.dump(debug_cfg, f, default_flow_style=False)
    print(f" -> Created Debug: {debug_filename}")

if __name__ == "__main__":
    generate_caltech_batch()
