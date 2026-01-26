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
        "info_dim": 20,
        "redundancy_factor": 2, 
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
        "log_interval": 50,
        "alignment_beta": 0.1 
    }
}

def generate_semi_caltech():
    output_dir = "configs/semi_caltech"
    os.makedirs(output_dir, exist_ok=True)
    
    latent_dims = [20, 40]
    redundancy_factors = [2, 5]
    
    count = 0
    for z in latent_dims:
        for L in redundancy_factors:
            model_configs = [
                # 1. Yang-Core: SMLVM Direct (GP Loss)
                {'name': f'yang_direct_Z{z}_L{L}', 'inf': 'direct', 'enc': 'mlp', 'ecc': 'none', 'cur_L': 1, 'gp': True},
                
                # 2. Yang-2025: SMLVM Amortized (GP Loss)
                {'name': f'yang_amortized_mlp_Z{z}_L{L}', 'inf': 'amortized', 'enc': 'mlp', 'ecc': 'none', 'cur_L': 1, 'gp': True},
                {'name': f'yang_amortized_cnn_Z{z}_L{L}', 'inf': 'amortized', 'enc': 'cnn', 'ecc': 'none', 'cur_L': 1, 'gp': True},

                # 3. Standard VAE: Uncoded (MSE Loss)
                {'name': f'vae_mlp_Z{z}_L{L}_uncoded', 'inf': 'amortized', 'enc': 'mlp', 'ecc': 'none', 'cur_L': 1, 'gp': False},
                
                # 4. Standard Coded VAE: (MSE Loss + ECC)
                {'name': f'vae_mlp_Z{z}_L{L}_rep', 'inf': 'amortized', 'enc': 'mlp', 'ecc': 'repetition', 'cur_L': L, 'gp': False},
                {'name': f'vae_mlp_Z{z}_L{L}_random', 'inf': 'amortized', 'enc': 'mlp', 'ecc': 'random_gaussian', 'cur_L': L, 'gp': False},
                
                # 5. Our Semi-Amortized (GP Loss + ECC)
                {'name': f'semi_mlp_Z{z}_L{L}_rep', 'inf': 'semi_amortized', 'enc': 'mlp', 'ecc': 'repetition', 'cur_L': L, 'gp': True},
                {'name': f'semi_mlp_Z{z}_L{L}_random', 'inf': 'semi_amortized', 'enc': 'mlp', 'ecc': 'random_gaussian', 'cur_L': L, 'gp': True},
                {'name': f'semi_cnn_Z{z}_L{L}_rep', 'inf': 'semi_amortized', 'enc': 'cnn', 'ecc': 'repetition', 'cur_L': L, 'gp': True},
                {'name': f'semi_cnn_Z{z}_L{L}_random', 'inf': 'semi_amortized', 'enc': 'cnn', 'ecc': 'random_gaussian', 'cur_L': L, 'gp': True},
            ]

            for c in model_configs:
                cfg = copy.deepcopy(BASE_CONFIG)
                cfg['experiment']['name'] = c['name']
                cfg['latent_space']['info_dim'] = z
                cfg['latent_space']['redundancy_factor'] = c['cur_L']
                cfg['latent_space']['ecc_type'] = c['ecc']
                cfg['model']['inference_mode'] = c['inf']
                cfg['model']['encoder_type'] = c['enc']
                cfg['training']['use_gp_loss'] = c['gp']
                
                filename = f"{output_dir}/{c['name']}.yaml"
                with open(filename, 'w') as f:
                    yaml.dump(cfg, f, default_flow_style=False)
                count += 1
                
    print(f"Generated {count} configurations in {output_dir}.")

if __name__ == "__main__":
    generate_semi_caltech()
