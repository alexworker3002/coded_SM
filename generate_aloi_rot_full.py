
import os
import yaml

def generate_rot_configs():
    output_dir = "configs/aloi_rot_full"
    os.makedirs(output_dir, exist_ok=True)
    
    # Common settings for Rotation Experiment
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
            # 3 Views for Rotation: Left(-5), Center(0), Right(+5)
            "view_left": {"input_dim": 82944, "likelihood": "gaussian"},
            "view_center": {"input_dim": 82944, "likelihood": "gaussian"},
            "view_right": {"input_dim": 82944, "likelihood": "gaussian"}
        },
        "dataset_kwargs": {
            "mode": "rotation" # New mode
        },
        "view_shapes": {
            "view_left": [3, 144, 192],
            "view_center": [3, 144, 192],
            "view_right": [3, 144, 192]
        }
    }

    experiments = []
    
    # Latent Dimensions to sweep
    z_dims = [128, 64, 32, 16, 8, 4]
    
    for z in z_dims:
        # 1. Baseline (Direct)
        experiments.append({"z": z, "mode": "direct", "enc": "mlp", "ecc": "none", "L": 1})
        
        # 2. Semi-Amortized (CNN2D)
        # For high Z (128, 64), L=2,4 is usually enough
        # For low Z (16, 8, 4), we need higher L (up to 8 or 16)
        if z >= 32:
            redundancies = [2, 4]
        else:
            redundancies = [2, 4, 8, 16]
            
        for L in redundancies:
            for ecc in ["random_gaussian", "repetition"]:
                experiments.append({"z": z, "mode": "semi_amortized", "enc": "cnn2d", "ecc": ecc, "L": L})

    for exp in experiments:
        z = exp["z"]
        mode = exp["mode"]
        enc = exp["enc"]
        ecc = exp["ecc"]
        L = exp["L"]
        
        # Construct filename
        if mode == "direct":
             exp_name = f"aloi_rot_z{z}_direct"
             filename = f"{exp_name}.yaml"
        else:
             ecc_short = "rand" if ecc == "random_gaussian" else "rep"
             exp_name = f"aloi_rot_z{z}_{enc}_semi_{ecc_short}_L{L}"
             filename = f"{exp_name}.yaml"

        # Create config copy
        cfg = common_config.copy()
        
        # Deep copy nested dicts to avoid mutation issues
        import copy
        cfg = copy.deepcopy(common_config)
        
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

        # Override LR for direct mode
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
    generate_rot_configs()
