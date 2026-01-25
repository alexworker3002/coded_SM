import os
import yaml
import copy

BASE_CONFIG = {
    "experiment": {
        "name": "template", 
        "device": "auto",  
        "seed": 42,
        "dataset": "mfeat"
    },
    "latent_space": {
        "info_dim": 10,
        "redundancy_factor": 1,
        "ecc_type": "none"
    },
    "model": {
        "inference_mode": "amortized",
        "encoder_type": "mlp"
    },
    "kernels": {
        "type": "ng_sm",
        "num_mixtures": 4,
        "rff_samples": 1000 
    },
    "views": {
        "fac":    { "input_dim": 216, "likelihood": "gaussian" },
        "fou":    { "input_dim": 76,  "likelihood": "gaussian" },
        "kar":    { "input_dim": 64,  "likelihood": "gaussian" },
        "pix":    { "input_dim": 240, "likelihood": "gaussian" },
        "zer":    { "input_dim": 47,  "likelihood": "gaussian" },
        "mor":    { "input_dim": 6,   "likelihood": "gaussian" }
    },
    "training": {
        "batch_size": 128, 
        "lr": 0.01,
        "epochs": 800, 
        "log_interval": 50,
        "alignment_beta": 0.1 # Default for semi
    }
}

def generate_semi_mfeat():
    output_dir = "configs/semi_mfeat"
    os.makedirs(output_dir, exist_ok=True)
    
    z = 10
    configs = []
    
    # 1. SMLVM (Baseline)
    configs.append({
        'name': f'semi_mfeat_smlvm_Z{z}',
        'z': z, 'L': 1, 'ecc': 'none', 'inf': 'direct', 'enc': 'mlp' 
    })
    
    # 2. VAE-MLP Uncoded (Baseline Amortized)
    configs.append({
        'name': f'semi_mfeat_vae_mlp_Z{z}_uncoded',
        'z': z, 'L': 1, 'ecc': 'none', 'inf': 'amortized', 'enc': 'mlp'
    })
    
    # 3. VAE-MLP Random L=5 (Existing best)
    configs.append({
        'name': f'semi_mfeat_vae_mlp_Z{z}_L5_random',
        'z': z, 'L': 5, 'ecc': 'random_gaussian', 'inf': 'amortized', 'enc': 'mlp'
    })
    
    # 4. [NEW] Semi-Amortized MLP Random L=5
    configs.append({
        'name': f'semi_mfeat_semi_mlp_Z{z}_L5_random',
        'z': z, 'L': 5, 'ecc': 'random_gaussian', 'inf': 'semi_amortized', 'enc': 'mlp'
    })
    
    # 5. [NEW] Semi-Amortized CNN Random L=5
    configs.append({
        'name': f'semi_mfeat_semi_cnn_Z{z}_L5_random',
        'z': z, 'L': 5, 'ecc': 'random_gaussian', 'inf': 'semi_amortized', 'enc': 'cnn'
    })

    print(f"Generating {len(configs)} configurations in {output_dir}...")
    
    for c in configs:
        cfg = copy.deepcopy(BASE_CONFIG)
        cfg['experiment']['name'] = c['name']
        cfg['latent_space']['info_dim'] = c['z']
        cfg['latent_space']['redundancy_factor'] = c['L']
        cfg['latent_space']['ecc_type'] = c['ecc']
        cfg['model']['inference_mode'] = c['inf']
        cfg['model']['encoder_type'] = c['enc']
        
        filename = f"{output_dir}/{c['name']}.yaml"
        with open(filename, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
        print(f" -> Created {filename}")

if __name__ == "__main__":
    generate_semi_mfeat()
