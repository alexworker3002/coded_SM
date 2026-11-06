
import os
import yaml

def generate_configs():
    output_dir = "configs/aloi_ablation_z16"
    os.makedirs(output_dir, exist_ok=True)
    
    # Common settings (Same as Z32/Z64)
    common_config = {
        "experiment": {
            "dataset": "aloi",
            "device": "auto",
            "seed": 42
        },
        "kernels": {
            "type": "ng_sm",
            "num_mixtures": 4,
            "rff_samples": 1000
        },
        "training": {
            "batch_size": 256,
            "epochs": 500,
            "lr": 0.001,
            "use_amp": True,
            "log_interval": 10,
            "alignment_beta": 0.01,
            "use_gp_loss": True,
            "num_workers": 0
        },
        "views": {
            "light_dir1": {"input_dim": 82944, "likelihood": "gaussian"},
            "light_dir2": {"input_dim": 82944, "likelihood": "gaussian"},
            "light_dir3": {"input_dim": 82944, "likelihood": "gaussian"},
            "light_dir4": {"input_dim": 82944, "likelihood": "gaussian"}
        },
        "dataset_kwargs": {
            "mode": "illumination"
        },
        "view_shapes": {
            "light_dir1": [3, 144, 192],
            "light_dir2": [3, 144, 192],
            "light_dir3": [3, 144, 192],
            "light_dir4": [3, 144, 192]
        }
    }

    # Experiments definition: Z=16
    experiments = [
        # Z = 16 (Extreme Compression)
        {"z": 16, "mode": "direct", "enc": "mlp", "ecc": "none", "L": 1},
        {"z": 16, "mode": "semi_amortized", "enc": "cnn2d", "ecc": "random_gaussian", "L": 2},
        {"z": 16, "mode": "semi_amortized", "enc": "cnn2d", "ecc": "random_gaussian", "L": 4},
        {"z": 16, "mode": "semi_amortized", "enc": "cnn2d", "ecc": "random_gaussian", "L": 8},
        {"z": 16, "mode": "semi_amortized", "enc": "cnn2d", "ecc": "repetition", "L": 2},
        {"z": 16, "mode": "semi_amortized", "enc": "cnn2d", "ecc": "repetition", "L": 4},
        {"z": 16, "mode": "semi_amortized", "enc": "cnn2d", "ecc": "repetition", "L": 8},
    ]

    for exp in experiments:
        z = exp["z"]
        mode = exp["mode"]
        enc = exp["enc"]
        ecc = exp["ecc"]
        L = exp["L"]
        
        # Construct filename and experiment name
        if mode == "direct":
             exp_name = f"aloi_z{z}_direct_full"
             filename = f"{exp_name}.yaml"
        else:
             ecc_short = "rand" if ecc == "random_gaussian" else "rep"
             exp_name = f"aloi_z{z}_{enc}_semi_{ecc_short}_L{L}"
             filename = f"{exp_name}.yaml"

        # Create config copy
        cfg = common_config.copy()
        cfg["experiment"] = common_config["experiment"].copy()
        cfg["experiment"]["name"] = exp_name
        
        cfg["latent_space"] = {
            "info_dim": z,
            "redundancy_factor": L,
            "ecc_type": ecc
        }
        
        cfg["model"] = {
            "inference_mode": mode,
            "encoder_type": enc
        }

        # Override LR for direct mode (0.01 vs 0.001)
        cfg["training"] = common_config["training"].copy()
        if mode == "direct":
            cfg["training"]["lr"] = 0.01 
        else:
            cfg["training"]["lr"] = 0.001

        # Write to file
        full_path = os.path.join(output_dir, filename)
        with open(full_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=None, sort_keys=False)
        print(f"Generated: {full_path}")

if __name__ == "__main__":
    generate_configs()
