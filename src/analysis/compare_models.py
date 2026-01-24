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
            
        # Infer redundancy from name
        if "L=5" in exp_name:
            redundancy = 5
            use_ecc = True
        elif "L=2" in exp_name and "Random" in exp_name: # Handle L=2 Random
            redundancy = 2
            use_ecc = True
        elif "Uncoded" in exp_name:
            redundancy = 1
            use_ecc = False
        else:
            # Default Coded L=2 (if not specified otherwise)
            redundancy = 2
            use_ecc = True
            
        # Infer ECC Mode (New for Path A)
        if "Random" in exp_name or "random" in ckpt_dir_prefix:
            ecc_mode = 'random_gaussian'
        else:
            ecc_mode = 'repetition'
            
        # Build Model
        model = CNG_MV_GPLVM(
            num_data=self.num_data,
            input_dim=10, 
            view_dims=self.view_dims,
            redundancy_factor=redundancy,
            use_ecc=use_ecc,
            num_mixtures=4,
            rff_samples=500,
            ecc_mode=ecc_mode 
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
                 # If Amortized, we need to pass data through encoder
                 # Use DataLoader to get all latents
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
        
    def plot_latent_space(self, latents_dict, method='tsne'):
        """
        Plot side-by-side comparison of latent spaces
        """
        num_models = len(latents_dict)
        fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 6))
        
        if num_models == 1:
            axes = [axes]
        
        for idx, (name, z) in enumerate(latents_dict.items()):
            print(f"Computing {method.upper()} for {name}...")
            if method == 'tsne':
                reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            else:
                reducer = PCA(n_components=2)
                
            z_2d = reducer.fit_transform(z)
            
            # Scatter plot
            ax = axes[idx]
            scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=self.labels, cmap='tab10', alpha=0.6, s=10)
            ax.set_title(f"{name} (Latent Space)")
            ax.axis('off')
            
        # Legend (use last ax)
        plt.colorbar(scatter, ax=axes, ticks=range(10), label='Digit Class')
        plt.suptitle(f"Latent Space Comparison: {method.upper()}")
        plt.savefig(f"comparison_latent_{method}.png")
        print(f"Saved latent plot to comparison_latent_{method}.png")

    def compute_class_wise_recon(self):
        # Result: {ModelName: {ClassID: {ViewName: Error}}}
        final_stats = {}
        
        # We need data loader to batch compute recon
        loader = DataLoader(self.dataset, batch_size=256, shuffle=False)
        
        for name in self.dirs.keys():
            print(f"Computing per-view errors for {name}...")
            model = self.load_model(name)
            
            # Accumulators
            # class_errors[class_id][view_name] = list of errors
            class_errors = {c: {v: [] for v in self.view_dims} for c in range(10)}
            
            with torch.no_grad():
                for views_batch, labels, indices in loader:
                    views_batch = {k: v.to(self.device).float() for k, v in views_batch.items()}
                    labels = labels.numpy()
                    
                    # Forward
                    # Handle indices vs views based on mode (handled inside forward now but be safe)
                    y_recons, _, _ = model.forward(indices.to(self.device), views_batch)
                    
                    # Compute error per sample per view
                    for v_name in self.view_dims:
                        # MSE per sample: (B, D) -> mean(dim=1) -> (B,)
                        mse_per_sample = (views_batch[v_name] - y_recons[v_name]).pow(2).mean(dim=1).cpu().numpy()
                        
                        for i, label in enumerate(labels):
                            class_errors[label][v_name].append(mse_per_sample[i])
                            
            # Aggregate mean
            mean_errors = {c: {v: np.mean(vals) for v, vals in view_dict.items()} 
                           for c, view_dict in class_errors.items()}
            final_stats[name] = mean_errors
            
        return final_stats
        
    def plot_radar_charts(self, stats):
        """
        Stats: {ModelName: {Class: {View: Error}}}
        """
        categories = list(self.view_dims.keys())
        N = len(categories)
        
        # Angles for radar chart
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        # Setup plot: 2 rows, 5 cols (for 10 classes)
        fig, axes = plt.subplots(2, 5, figsize=(20, 9), subplot_kw=dict(polar=True))
        axes = axes.flatten()
        
        for c in range(10):
            ax = axes[c]
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            
            plt.xticks(angles[:-1], categories)
            ax.set_rlabel_position(0)
            
            # Plot each model
            # Colors for 4 models
            colors = {
                'Uncoded': 'red', 
                'Coded (L=2)': 'blue',
                'Coded (L=5)': 'green',
                'Coded (L=5, Random)': 'purple',
                'Coded (L=2, Random)': 'orange'
            }
            
            for model_name, model_data in stats.items():
                values = [model_data[c][cat] for cat in categories]
                values += values[:1]
                
                # Check color
                color = colors.get(model_name, 'black')
                
                ax.plot(angles, values, linewidth=1, linestyle='solid', label=model_name, color=color)
                ax.fill(angles, values, color=color, alpha=0.05) # lighter fill
                
            ax.set_title(f"Class {c}", size=11, weight='bold', y=1.1)
            
        # Legend (take from first axis)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        plt.tight_layout()
        plt.savefig("comparison_radar_recon.png")
        print("Saved radar chart to comparison_radar_recon.png")

    def run(self):
        # 1. Latent Space
        latents = self.extract_latents()
        self.plot_latent_space(latents, method='tsne')
        
        # 2. Redisual/Recon Radar
        recon_stats = self.compute_class_wise_recon()
        self.plot_radar_charts(recon_stats)
    
    def plot_aggregated_metrics(self):
         # Placeholder for functionality defined in task boundaries
         pass

if __name__ == "__main__":
    # Define experiment mapping
    # Define experiment mapping
    experiments = {
        "Uncoded": "cng_mvlvm_mfeat_uncoded",
        "Coded (L=2)": "cng_mvlvm_mfeat_trial_01",
        "Coded (L=5)": "cng_mvlvm_mfeat_redundancy_5",
        "Coded (L=5, Random)": "cng_local_L5_random_gaussian",
        "Coded (L=2, Random)": "cng_local_L2_random_gaussian"
    }
    
    comparator = ModelComparator(
        exp_dirs=experiments,
        device='cpu'
    )
    comparator.run()
