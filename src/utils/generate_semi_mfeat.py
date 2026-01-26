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
        "fac": { "input_dim": 216, "likelihood": "gaussian" },
        "fou": { "input_dim": 76,  "likelihood": "gaussian" },
        "kar": { "input_dim": 64,  "likelihood": "gaussian" },
        "pix": { "input_dim": 240, "likelihood": "gaussian" },
        "zer": { "input_dim": 47,  "likelihood": "gaussian" },
        "mor": { "input_dim": 6,   "likelihood": "gaussian" }
    },
    "training": {
        "batch_size": 128, 
        "lr": 0.01,
        "epochs": 500, 
        "log_interval": 50,
        "alignment_beta": 0.1,
        "use_gp_loss": True
    }
}

def generate_semi_mfeat(output_root="configs/semi_mfeat"):
    os.makedirs(output_root, exist_ok=True)
    
    # Target: Z=10
    latent_dims = [10]
    # L=[2, 5] as requested
    redundancy_factors = [2, 5]
    
    count = 0
    for z in latent_dims:
        for L in redundancy_factors:
            model_configs = [
                # 1. Yang: SMLVM Direct (GP Loss) - Baseline
                # One baseline per (Z) but we generate for loop simplicity
                {'name': f'yang_direct_Z{z}_L{L}', 'inf': 'direct', 'enc': 'mlp', 'ecc': 'none', 'cur_L': 1, 'gp': True},
                
                # 2. Coded MLP VAE (Amortized, MSE Loss + ECC)
                # Note: User said "coded mlp vae", usually implies MSE loss, not GP.
                # Assuming "Coded VAE" refers to our previous baseline: Amortized + ECC + MSE
                {'name': f'vae_mlp_Z{z}_L{L}_rep', 'inf': 'amortized', 'enc': 'mlp', 'ecc': 'repetition', 'cur_L': L, 'gp': False},
                {'name': f'vae_mlp_Z{z}_L{L}_random', 'inf': 'amortized', 'enc': 'mlp', 'ecc': 'random_gaussian', 'cur_L': L, 'gp': False},
                
                # 3. Semi-Amortized Random (GP Loss + ECC)
                {'name': f'semi_mlp_Z{z}_L{L}_random', 'inf': 'semi_amortized', 'enc': 'mlp', 'ecc': 'random_gaussian', 'cur_L': L, 'gp': True},
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
                
                filename = f"{output_root}/{c['name']}.yaml"
                with open(filename, 'w') as f:
                    yaml.dump(cfg, f, default_flow_style=False)
                count += 1
                
    print(f"Generated {count} configurations in {output_root}.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="configs/semi_mfeat")
    args = parser.parse_args()
    generate_semi_mfeat(args.output_dir)
