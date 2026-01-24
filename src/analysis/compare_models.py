# src/analysis/compare_models.py

import os
# Fix for Mac OpenMP duplicate library error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import glob
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Local Imports
from src.models.cng_model import CNG_MV_GPLVM
from src.utils.data_utils import load_mfeat_data

# Set style
sns.set(style="whitegrid")

class ModelComparator:
    def __init__(self, exp_dirs, device='cpu'):
        """
        exp_dirs: Dict {Display Name: Checkpoint Dir Prefix}
        """
        self.device = device
        self.dirs = exp_dirs
        
        # Load Data once
        self.dataset = load_mfeat_data(mode="real")
        self.labels = self.dataset.labels.numpy()
        self.view_dims = {k: v.shape[1] for k, v in self.dataset.views.items()}
        self.num_data = len(self.dataset)
        
    def load_model(self, exp_name):
        ckpt_dir_prefix = self.dirs[exp_name]
        
        # Find path
        candidates = glob.glob(f"checkpoints/{ckpt_dir_prefix}*")
        if not candidates:
            # Fallback for full paths or relative paths
            candidates = glob.glob(f"{ckpt_dir_prefix}*")
            if not candidates:
                raise ValueError(f"No checkpoint found for prefix {ckpt_dir_prefix}")
        
        actual_dir = sorted(candidates)[-1] # Take latest
        print(f"[{exp_name}] Using run: {actual_dir}")
            
        # Infer redundancy and ECC Mode
        if "Uncoded" in exp_name or "L=1" in exp_name:
            redundancy = 1
            use_ecc = False
            ecc_mode = 'repetition' # Dummy
        else:
            use_ecc = True
            if "L=10" in exp_name:
                redundancy = 10
            elif "L=5" in exp_name:
                redundancy = 5
            elif "L=2" in exp_name:
                redundancy = 2
            else:
                redundancy = 5 # Default
            
            if "Random" in exp_name or "random" in exp_name:
                ecc_mode = 'random_gaussian'
            else:
                ecc_mode = 'repetition'

        # Infer info_dim (Z) from name
        if "Z=2" in exp_name or "Z2" in exp_name:
            input_dim = 2
        elif "Z=5" in exp_name or "Z5" in exp_name:
            input_dim = 5
        elif "Z=10" in exp_name or "Z10" in exp_name:
            input_dim = 10
        else:
            input_dim = 2 # Default

        # Infer inference_mode & encoder_type
        if "VAE" in exp_name or "vae" in exp_name:
            inference_mode = 'amortized'
            if "CNN" in exp_name or "cnn" in exp_name:
                encoder_type = 'cnn'
            else:
                encoder_type = 'mlp'
        else:
            inference_mode = 'direct'
            encoder_type = 'mlp'
            
        # Build Model
        model = CNG_MV_GPLVM(
            num_data=self.num_data,
            input_dim=input_dim,
            view_dims=self.view_dims,
            redundancy_factor=redundancy,
            use_ecc=use_ecc,
            num_mixtures=4,
            rff_samples=500,
            ecc_mode=ecc_mode,
            inference_mode=inference_mode,
            encoder_type=encoder_type
        ).to(self.device)
        
        # Load Weights
        model_path = os.path.join(actual_dir, "final_model.pth")
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model file not found at {model_path}")
             
        state_dict = torch.load(model_path, map_location=self.device)
        
        # Compatibility Fix
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"⚠️ Warning loading {exp_name}: {e}")
            
        model.eval()
        
        return model
        
    def extract_latents(self):
        results = {}
        for name in self.dirs.keys():
            print(f"Extracting latents for {name}...")
            model = self.load_model(name)
            
            # Handling Amortized Inference (Encoder) vs Direct
            if model.inference_mode == 'direct':
                z_mu = model.q_mu.detach().cpu().numpy()
            else:
                 loader = DataLoader(self.dataset, batch_size=256, shuffle=False)
                 z_list = []
                 with torch.no_grad():
                     for views, _, _ in loader:
                         views = {k: v.to(self.device).float() for k, v in views.items()}
                         mu, _ = model.encoder(views)
                         z_list.append(mu.cpu())
                 z_mu = torch.cat(z_list, dim=0).numpy()

            results[name] = z_mu
        return results
        
    def run_group(self, group_name, experiments_subset):
        print(f"\n=== Analyzing Group: {group_name} ===")
        # Filter dirs for this group
        self.dirs = experiments_subset
        
        # 1. Latent Space
        latents = self.extract_latents()
        self.plot_latent_space(latents, method='tsne', suffix=f"_{group_name}")
        
        # 2. Redisual/Recon Radar
        recon_stats = self.compute_class_wise_recon()
        self.plot_radar_charts(recon_stats, suffix=f"_{group_name}")
        
        # 3. Loss Curves
        loss_data = self.extract_loss_history()
        if loss_data:
            self.plot_loss_curves(loss_data, suffix=f"_{group_name}")

    def plot_latent_space(self, latents_dict, method='tsne', suffix=""):
        num_models = len(latents_dict)
        if num_models == 0: return

        fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 6))
        if num_models == 1: axes = [axes]
        
        for idx, (name, z) in enumerate(latents_dict.items()):
            print(f"Computing {method.upper()} for {name}...")
            if method == 'tsne':
                reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            else:
                reducer = PCA(n_components=2)
            z_2d = reducer.fit_transform(z)
            
            ax = axes[idx]
            scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=self.labels, cmap='tab10', alpha=0.6, s=10)
            ax.set_title(f"{name}")
            ax.axis('off')
            
        plt.colorbar(scatter, ax=axes, ticks=range(10), label='Digit Class')
        plt.suptitle(f"Latent Space Comparison {suffix}: {method.upper()}")
        plt.savefig(f"comparison_latent_{method}{suffix}.png")
        print(f"Saved plot: comparison_latent_{method}{suffix}.png")

    def compute_class_wise_recon(self):
        # Result: {ModelName: {ClassID: {ViewName: Error}}}
        final_stats = {}
        loader = DataLoader(self.dataset, batch_size=256, shuffle=False)
        
        for name in self.dirs.keys():
            print(f"Computing per-view errors for {name}...")
            model = self.load_model(name)
            
            # Accumulators
            class_errors = {c: {v: [] for v in self.view_dims} for c in range(10)}
            
            with torch.no_grad():
                for views_batch, labels, indices in loader:
                    views_batch = {k: v.to(self.device).float() for k, v in views_batch.items()}
                    labels = labels.numpy()
                    
                    y_recons, _, _ = model.forward(indices.to(self.device), views_batch)
                    
                    for v_name in self.view_dims:
                        mse_per_sample = (views_batch[v_name] - y_recons[v_name]).pow(2).mean(dim=1).cpu().numpy()
                        for i, label in enumerate(labels):
                            class_errors[label][v_name].append(mse_per_sample[i])
                            
            mean_errors = {c: {v: np.mean(vals) for v, vals in view_dict.items()} 
                           for c, view_dict in class_errors.items()}
            final_stats[name] = mean_errors
            
        return final_stats

    def plot_radar_charts(self, stats, suffix=""):
        categories = list(self.view_dims.keys())
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 9), subplot_kw=dict(polar=True))
        axes = axes.flatten()
        
        def get_color(name):
            if 'SMLVM' in name: return 'black'
            if 'Random' in name: return 'magenta'
            if 'Rep' in name: return 'cyan'
            return 'gray'

        for c in range(10):
            ax = axes[c]
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            plt.xticks(angles[:-1], categories)
            ax.set_rlabel_position(0)
            
            for model_name, model_data in stats.items():
                values = [model_data[c][cat] for cat in categories]
                values += values[:1]
                color = get_color(model_name)
                ax.plot(angles, values, linewidth=1, linestyle='solid', label=model_name, color=color)
                ax.fill(angles, values, color=color, alpha=0.05)
                
            ax.set_title(f"Class {c}", size=11, weight='bold', y=1.1)
            
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        plt.tight_layout()
        plt.savefig(f"comparison_radar_recon{suffix}.png")
        print(f"Saved radar: comparison_radar_recon{suffix}.png")

    def get_log_dir(self, exp_name):
        ckpt_dir_prefix = self.dirs[exp_name]
        candidates = glob.glob(f"logs/{ckpt_dir_prefix}*")
        if not candidates:
            # Try recursive or exact match
            candidates = glob.glob(f"{ckpt_dir_prefix}*") 
            if not candidates:
                 print(f"⚠️ Warning: No logs found for {exp_name}")
                 return None
        return sorted(candidates)[-1]

    def extract_loss_history(self):
        loss_data = {}
        for name in self.dirs.keys():
            log_dir = self.get_log_dir(name)
            if not log_dir:
                continue
            try:
                ea = EventAccumulator(log_dir)
                ea.Reload()
                tags = ea.Tags()['scalars']
                if 'Loss/Total' in tags:
                    events = ea.Scalars('Loss/Total')
                    steps = [e.step for e in events]
                    values = [e.value for e in events]
                    loss_data[name] = (steps, values)
            except Exception as e:
                print(f"⚠️ Error reading logs for {name}: {e}")
        return loss_data

    def plot_loss_curves(self, loss_data, suffix=""):
        plt.figure(figsize=(10, 6))
        def get_color(name):
            if 'SMLVM' in name: return 'black'
            if 'Random' in name: return 'magenta'
            if 'Rep' in name: return 'cyan'
            return 'gray'
            
        for name, (steps, values) in loss_data.items():
            color = get_color(name)
            plt.plot(steps, values, label=name, color=color, alpha=0.8, linewidth=1.5)
            
        plt.xlabel('Epoch')
        plt.ylabel('Loss (ELBO)')
        plt.title(f'Training Loss Comparison {suffix}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"comparison_loss_curves{suffix}.png")
        print(f"Saved loss plot: comparison_loss_curves{suffix}.png")

if __name__ == "__main__":
    # Define 3 Groups of Experiments
    groups = {
        "Z2": {
            "SMLVM (Z=2)": "smlvm_Z2_L1",
            "VAE-Rep (Z=2)": "vae_mlp_Z2_L2_rep",
            "VAE-Random (Z=2)": "vae_mlp_Z2_L2_random"
        },
        "Z5": {
            "SMLVM (Z=5)": "smlvm_Z5_L1",
            "VAE-Rep (Z=5)": "vae_mlp_Z5_L2_rep",
            "VAE-Random (Z=5)": "vae_mlp_Z5_L2_random"
        },
        "Z10": {
            "SMLVM (Z=10)": "smlvm_Z10_L1",
            "VAE-Rep (Z=10)": "vae_mlp_Z10_L2_rep",
            "VAE-Random (Z=10)": "vae_mlp_Z10_L2_random"
        }
    }
    
    # Run analysis for each group separately
    for group_name, experiments_map in groups.items():
        comparator = ModelComparator(
            exp_dirs=experiments_map, # Pass only this group's subset
            device='cpu'
        )
        comparator.run_group(group_name, experiments_map)
