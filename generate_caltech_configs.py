import yaml
import os

os.makedirs("configs/caltech_batch", exist_ok=True)

# Caltech view info
caltech_views = {
    "wm": {"input_dim": 64, "likelihood": "gaussian"},
    "fac": {"input_dim": 448, "likelihood": "gaussian"},
    "fou": {"input_dim": 76, "likelihood": "gaussian"},
    "kar": {"input_dim": 64, "likelihood": "gaussian"},
    "pix": {"input_dim": 240, "likelihood": "gaussian"},
    "zer": {"input_dim": 47, "likelihood": "gaussian"}
}

def create_config(name, z_dim, L, ecc):
    cfg = {
        'experiment': {
            'dataset': 'caltech',
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
            'encoder_type': 'mlp',
            'inference_mode': 'semi_amortized'
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
        'views': caltech_views
    }
    with open(f"configs/caltech_batch/{name}.yaml", 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

# Dimensions: 10, 20, 40
for z in [10, 20, 40]:
    # Baseline (Redundancy 1, no ECC)
    create_config(f"caltech_z{z}_baseline", z, 1, "none")
    
    # Coded (Redundancy 2)
    for ecc_type, short in [('repetition', 'rep'), ('random_gaussian', 'rand')]:
        create_config(f"caltech_z{z}_L2_{short}", z, 2, ecc_type)

print("Generated configs in configs/caltech_batch/:")
for f in sorted(os.listdir("configs/caltech_batch")):
    print(f"  {f}")
