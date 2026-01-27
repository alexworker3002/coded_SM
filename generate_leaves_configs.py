import yaml
import os

dest_dir = "configs/leaves_complex"
os.makedirs(dest_dir, exist_ok=True)

leaves_views = {
    "shape": {"input_dim": 64, "likelihood": "gaussian"},
    "texture": {"input_dim": 64, "likelihood": "gaussian"},
    "margin": {"input_dim": 64, "likelihood": "gaussian"}
}

def create_config(name, z_dim, L, ecc, encoder, mode='semi_amortized'):
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
            'encoder_type': encoder,
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

# Dimensions: 32, 64
for z in [32, 64]:
    # 1. YANG Baseline (Pure Amortized MLP - the GOLD standard base)
    create_config(f"leaves_z{z}_yang_baseline", z, 1, "none", "mlp", mode='amortized')
    
    # 2. Research Group: MLP Semi-Amortized variants
    create_config(f"leaves_z{z}_mlp_semi_base", z, 1, "none", "mlp", mode='semi_amortized')
    for L in [2, 5]:
        for ecc_type, short in [('repetition', 'rep'), ('random_gaussian', 'rand')]:
            create_config(f"leaves_z{z}_mlp_L{L}_{short}", z, L, ecc_type, "mlp")
            
    # 3. Research Group: CNN Semi-Amortized variants (To test architecture benefit)
    create_config(f"leaves_z{z}_cnn_semi_base", z, 1, "none", "cnn", mode='semi_amortized')
    for L in [2, 5]:
        for ecc_type, short in [('repetition', 'rep'), ('random_gaussian', 'rand')]:
            create_config(f"leaves_z{z}_cnn_L{L}_{short}", z, L, ecc_type, "cnn")

print("Generated 22 configs in configs/leaves_complex/")
print("- 2 x Yang Baseline (Pure MLP)")
print("- 10 x MLP Semi Group")
print("- 10 x CNN Semi Group")
