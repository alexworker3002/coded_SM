import yaml
import os

dest_dir = "configs/leaves_variants"
os.makedirs(dest_dir, exist_ok=True)

leaves_views = {
    "shape": {"input_dim": 64, "likelihood": "gaussian"},
    "texture": {"input_dim": 64, "likelihood": "gaussian"},
    "margin": {"input_dim": 64, "likelihood": "gaussian"}
}

def create_config(name, z_dim, L, ecc, encoder, mode):
    cfg = {
        'experiment': {
            'dataset': '100leaves',
            'device': 'auto',
            'name': name,
            'seed': 42
        },
        'kernels': {
            'num_mixtures': 4,
            'rff_samples': 1000,
            'type': 'ng_sm'
        },
        'latent_space': {
            'ecc_type': ecc,
            'info_dim': z_dim,
            'redundancy_factor': L
        },
        'model': {
            'encoder_type': encoder if encoder else 'mlp',
            'inference_mode': mode
        },
        'training': {
            'alignment_beta': 0.01,
            'batch_size': 512,
            'epochs': 500,
            'log_interval': 10,
            'lr': 0.01,
            'use_amp': False,
            'use_gp_loss': True
        },
        'views': leaves_views
    }
    with open(f"{dest_dir}/{name}.yaml", 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

# Latent Dimensions
for Z in [16, 32, 64]:
    
    # Redundancy config based on Z
    redundancies = [2, 4] if Z == 16 else [2]

    # 1. Baseline: Direct Optimization (Standard GPLVM)
    create_config(f"leaves_z{Z}_direct", Z, 1, "none", "mlp", "direct")

    # 2. Baseline: Amortized (VAE) - No Redundancy
    for arch in ['mlp', 'cnn']:
        create_config(f"leaves_z{Z}_{arch}_amortized_base", Z, 1, "none", arch, "amortized")

    # 3. Baseline: Semi-Amortized - No Redundancy
    for arch in ['mlp', 'cnn']:
        create_config(f"leaves_z{Z}_{arch}_semi_base", Z, 1, "none", arch, "semi_amortized")

    # 4. Coded Variants (Coded-Amortized & Coded-Semi)
    for arch in ['mlp', 'cnn']:
        for mode_short, mode_val in [('amortized', 'amortized'), ('semi', 'semi_amortized')]:
            for L in redundancies:
                for ecc_type, ecc_short in [('repetition', 'rep'), ('random_gaussian', 'rand')]:
                    name = f"leaves_z{Z}_{arch}_{mode_short}_{ecc_short}_L{L}"
                    create_config(name, Z, L, ecc_type, arch, mode_val)

print("Generated configs in configs/leaves_variants/")
