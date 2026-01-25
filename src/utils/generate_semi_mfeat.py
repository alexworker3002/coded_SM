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
        "redundancy_factor": 2, # Changed to L=2
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
        "alignment_beta": 0.1 
    }
}

def generate_semi_mfeat():
    output_dir = "configs/semi_mfeat"
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean up old configs if needed, but here we just overwrite
    z = 10
    L = 2 # Redundancy factor
    
    model_configs = [
        # 1. Baseline: Direct
        {'name': f'semi_mfeat_smlvm_Z{z}', 'inf': 'direct', 'enc': 'mlp', 'ecc': 'none', 'L': 1},
        
        # 2. Baseline: Amortized Uncoded
        {'name': f'semi_mfeat_vae_mlp_Z{z}_uncoded', 'inf': 'amortized', 'enc': 'mlp', 'ecc': 'none', 'L': 1},
        {'name': f'semi_mfeat_vae_cnn_Z{z}_uncoded', 'inf': 'amortized', 'enc': 'cnn', 'ecc': 'none', 'L': 1},
        
        # 3. Amortized Coded (L=2)
        {'name': f'semi_mfeat_vae_mlp_Z{z}_L2_rep', 'inf': 'amortized', 'enc': 'mlp', 'ecc': 'repetition', 'L': L},
        {'name': f'semi_mfeat_vae_mlp_Z{z}_L2_random', 'inf': 'amortized', 'enc': 'mlp', 'ecc': 'random_gaussian', 'L': L},
        {'name': f'semi_mfeat_vae_cnn_Z{z}_L2_rep', 'inf': 'amortized', 'enc': 'cnn', 'ecc': 'repetition', 'L': L},
        {'name': f'semi_mfeat_vae_cnn_Z{z}_L2_random', 'inf': 'amortized', 'enc': 'cnn', 'ecc': 'random_gaussian', 'L': L},
        
        # 4. Semi-Amortized Coded (L=2)
        {'name': f'semi_mfeat_semi_mlp_Z{z}_L2_rep', 'inf': 'semi_amortized', 'enc': 'mlp', 'ecc': 'repetition', 'L': L},
        {'name': f'semi_mfeat_semi_mlp_Z{z}_L2_random', 'inf': 'semi_amortized', 'enc': 'mlp', 'ecc': 'random_gaussian', 'L': L},
        {'name': f'semi_mfeat_semi_cnn_Z{z}_L2_rep', 'inf': 'semi_amortized', 'enc': 'cnn', 'ecc': 'repetition', 'L': L},
        {'name': f'semi_mfeat_semi_cnn_Z{z}_L2_random', 'inf': 'semi_amortized', 'enc': 'cnn', 'ecc': 'random_gaussian', 'L': L},
    ]

    print(f"Generating {len(model_configs)} configurations in {output_dir}...")
    
    for c in model_configs:
        cfg = copy.deepcopy(BASE_CONFIG)
        cfg['experiment']['name'] = c['name']
        cfg['latent_space']['info_dim'] = z
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
