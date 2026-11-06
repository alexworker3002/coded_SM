import yaml
import os

dest_dir = "configs/mfeat_variants"
os.makedirs(dest_dir, exist_ok=True)

# MFeat Views (Standard)
mfeat_views = {
    "fac": {"input_dim": 216, "likelihood": "gaussian"},
    "fou": {"input_dim": 76, "likelihood": "gaussian"},
    "kar": {"input_dim": 64, "likelihood": "gaussian"},
    "pix": {"input_dim": 240, "likelihood": "gaussian"},
    "zer": {"input_dim": 47, "likelihood": "gaussian"},
    "mor": {"input_dim": 6, "likelihood": "gaussian"}
}

def create_config(name, z_dim, L, ecc, encoder, mode):
    cfg = {
        'experiment': {
            'dataset': 'mfeat',
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
            'encoder_type': encoder if encoder else 'mlp', # Default for direct
            'inference_mode': mode
        },
        'training': {
            'alignment_beta': 0.01,
            'batch_size': 256, # Smaller batch due to 2000 samples
            'epochs': 500,
            'log_interval': 10,
            'lr': 0.01,
            'use_amp': False,
            'use_gp_loss': True
        },
        'views': mfeat_views
    }
    with open(f"{dest_dir}/{name}.yaml", 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

Z = 20

# 1. Baseline: Direct Optimization (Standard GPLVM)
#    No Encoder, No ECC.
create_config(f"mfeat_z{Z}_direct", Z, 1, "none", "mlp", "direct")

# 2. Baseline: Amortized (VAE) - No Redundancy
for arch in ['mlp', 'cnn']:
    create_config(f"mfeat_z{Z}_{arch}_amortized_base", Z, 1, "none", arch, "amortized")

# 3. Baseline: Semi-Amortized - No Redundancy
for arch in ['mlp', 'cnn']:
    create_config(f"mfeat_z{Z}_{arch}_semi_base", Z, 1, "none", arch, "semi_amortized")

# 4. Coded Variants (Redundancy 2 & 5)
#    Combinations: Arch x Mode[amortized/semi] x ECC[rep/rand] x L[2/5]
for arch in ['mlp', 'cnn']:
    for mode_short, mode_val in [('amortized', 'amortized'), ('semi', 'semi_amortized')]:
        for L in [2, 5]:
            for ecc_type, ecc_short in [('repetition', 'rep'), ('random_gaussian', 'rand')]:
                name = f"mfeat_z{Z}_{arch}_{mode_short}_{ecc_short}_L{L}"
                create_config(name, Z, L, ecc_type, arch, mode_val)

print("Generated 21 configs in configs/mfeat_variants/")
