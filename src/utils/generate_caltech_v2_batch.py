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
        "gabor":    { "input_dim": 48,   "likelihood": "gaussian" },
        "wm":       { "input_dim": 40,   "likelihood": "gaussian" },
        "centrist": { "input_dim": 254,  "likelihood": "gaussian" },
        "hog":      { "input_dim": 1984, "likelihood": "gaussian" },
        "gist":     { "input_dim": 512,  "likelihood": "gaussian" },
        "lbp":      { "input_dim": 928,  "likelihood": "gaussian" }
    },
    "training": {
        "batch_size": 128, 
        "lr": 0.01,
        "epochs": 800, 
        "log_interval": 50
    }
}

def generate_caltech_v2_batch():
    output_dir = "configs/caltech_v2_batch"
    os.makedirs(output_dir, exist_ok=True)
    
    dims = [10, 20, 40, 80]
    configs = []
    
    for z in dims:
        # Prefix everything with caltech_v2 to ensure unique folders
        configs.append({
            'name': f'caltech_v2_smlvm_Z{z}_L1',
            'z': z, 'L': 1, 'ecc': 'none', 'inf': 'direct', 'enc': 'mlp' 
        })
        configs.append({
            'name': f'caltech_v2_vae_mlp_Z{z}_L1_uncoded',
            'z': z, 'L': 1, 'ecc': 'none', 'inf': 'amortized', 'enc': 'mlp'
        })
        configs.append({
            'name': f'caltech_v2_vae_cnn_Z{z}_L1_uncoded',
            'z': z, 'L': 1, 'ecc': 'none', 'inf': 'amortized', 'enc': 'cnn'
        })
        configs.append({
            'name': f'caltech_v2_vae_mlp_Z{z}_L2_rep',
            'z': z, 'L': 2, 'ecc': 'repetition', 'inf': 'amortized', 'enc': 'mlp'
        })
        configs.append({
            'name': f'caltech_v2_vae_cnn_Z{z}_L2_rep',
            'z': z, 'L': 2, 'ecc': 'repetition', 'inf': 'amortized', 'enc': 'cnn'
        })
        configs.append({
            'name': f'caltech_v2_vae_mlp_Z{z}_L5_random',
            'z': z, 'L': 5, 'ecc': 'random_gaussian', 'inf': 'amortized', 'enc': 'mlp'
        })
        configs.append({
            'name': f'caltech_v2_vae_cnn_Z{z}_L5_random',
            'z': z, 'L': 5, 'ecc': 'random_gaussian', 'inf': 'amortized', 'enc': 'cnn'
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
    generate_caltech_v2_batch()
