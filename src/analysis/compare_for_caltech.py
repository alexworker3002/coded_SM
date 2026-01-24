import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob

# Local Imports
from src.models.cng_model import CNG_MV_GPLVM
from src.utils.data_caltech import load_caltech_data

# Set Style
sns.set(style="whitegrid", context="talk")

def cluster_acc(y_true, y_pred):
    """
    Compute Clustering Accuracy using Hungarian Algorithm
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size

class AdvancedComparator:
    def __init__(self, exp_dirs, device='cpu'):
        self.device = device
        self.exp_dirs = exp_dirs
        
        # Load Data (Once)
        print("Loading Caltech101-7 Data...")
        try:
             # Ensure we use real data if possible, script will handle mock fallback internally
             self.dataset = load_caltech_data()
        except:
             # Fallback if import fails or something, though data_caltech handles it
             from src.utils.data_caltech import load_caltech_data
             self.dataset = load_caltech_data()

        self.loader = DataLoader(self.dataset, batch_size=512, shuffle=False)
        self.num_data = len(self.dataset)
        self.view_dims = {k: v.shape[1] for k, v in self.dataset.views.items()}
        self.labels = self.dataset.labels.numpy()
        self.num_classes = len(np.unique(self.labels))
        print(f"Data Loaded: N={self.num_data}, Classes={self.num_classes}")

    def load_model(self, exp_name):
        """
        Load model weights by inferring params from experiment name.
        """
        ckpt_prefix = self.exp_dirs[exp_name]
        
        # Find path
        candidates = glob.glob(f"checkpoints/{ckpt_prefix}*")
        if not candidates:
            candidates = glob.glob(f"{ckpt_prefix}*")
            if not candidates:
                raise ValueError(f"No checkpoint found for prefix {ckpt_prefix}")
        
        actual_dir = sorted(candidates)[-1]
        print(f"[{exp_name}] Loading from: {actual_dir}")

        # --- Parse Params from Name ---
        # Format: caltech_[algo]_Z[z]_[...options]
        # Example: caltech_vae_cnn_Z10_L5_random
        
        name_lower = ckpt_prefix.lower()
        
        # Infomation Dim (Z)
        if "z10" in name_lower:
            input_dim = 10
        elif "z5" in name_lower:
            input_dim = 5
        elif "z2" in name_lower:
            input_dim = 2
        else:
            input_dim = 5 # Default
            
        # Redundancy (L)
        if "l10" in name_lower:
            redundancy = 10
        elif "l5" in name_lower:
            redundancy = 5
        elif "l2" in name_lower:
            redundancy = 2
        else:
            redundancy = 1
            
        # ECC Mode
        if "random" in name_lower:
            ecc_mode = 'random_gaussian'
        elif "rep" in name_lower:
            ecc_mode = 'repetition'
        else:
            ecc_mode = 'repetition' # default/dummy
            
        # Use ECC?
        use_ecc = True
        if "uncoded" in name_lower or "l1" in name_lower or "smlvm" in name_lower:
            if redundancy == 1:
                use_ecc = False
        
        # Inference Mode & Encoder
        if "vae" in name_lower:
            inference_mode = 'amortized'
            if "cnn" in name_lower:
                encoder_type = 'cnn'
            else:
                encoder_type = 'mlp'
        else:
            inference_mode = 'direct'
            encoder_type = 'mlp'
            
        # Init Model
        model = CNG_MV_GPLVM(
            num_data=self.num_data,
            input_dim=input_dim,
            view_dims=self.view_dims,
            redundancy_factor=redundancy,
            use_ecc=use_ecc,
            ecc_mode=ecc_mode,
            inference_mode=inference_mode,
            encoder_type=encoder_type,
            rff_samples=1000 # Matches Caltech Config
        ).to(self.device)
        
        # Load Weights
        model_path = os.path.join(actual_dir, "final_model.pth")
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"⚠️ Warning loading weights: {e}")
            
        model.eval()
        return model

    def get_latents(self, model):
        """Extract Z_mu for all data"""
        z_list = []
        
        # If Direct Optimization (SMLVM), Z is a parameter
        if model.inference_mode == 'direct':
            return model.q_mu.detach().cpu().numpy()
            
        # If Amortized, pass data through Encoder
        with torch.no_grad():
            for views, _, _ in self.loader:
                views = {k: v.to(self.device).float() for k, v in views.items()}
                mu, _ = model.encoder(views)
                z_list.append(mu.cpu())
        return torch.cat(z_list, dim=0).numpy()

    def run_clustering_test(self, z_data):
        """Compute NMI and ACC"""
        kmeans = KMeans(n_clusters=self.num_classes, n_init=20, random_state=42)
        y_pred = kmeans.fit_predict(z_data)
        
        acc = cluster_acc(self.labels, y_pred)
        nmi = normalized_mutual_info_score(self.labels, y_pred)
        return acc, nmi

    def run_robustness_test(self, model, full_z):
        """
        Compute Latent Shift when masking 50% of views.
        Since Caltech has 6 views, we mask 3.
        """
        # Models with Direct Inference (SMLVM) cannot do 'inference' on new/masked data easily
        # without retraining or optimization.
        if model.inference_mode == 'direct':
            return np.nan # Not applicable
            
        shift_dist = 0
        count = 0
        
        # Drop first 3 views: Gabor, WM, Centrist
        # Keep: HOG, GIST, LBP
        drop_views = ['gabor', 'wm', 'centrist'] 
        
        with torch.no_grad():
            for i, (views, _, _) in enumerate(self.loader):
                views = {k: v.to(self.device).float() for k, v in views.items()}
                batch_size = list(views.values())[0].shape[0]
                
                # Masking: Replace dropped views with Zeros
                # Since data is standardized, 0 is the mean. Correct approach.
                masked_views = {k: (v if k not in drop_views else torch.zeros_like(v)) 
                                for k, v in views.items()}
                
                # Infer Z_partial
                # Ensure model is in eval mode for BatchNorm
                model.eval() 
                z_partial, _ = model.encoder(masked_views)
                
                # Get Z_full (from precomputed array)
                start_idx = i * self.loader.batch_size
                z_full_batch = torch.tensor(full_z[start_idx : start_idx+batch_size]).to(self.device)
                
                # Euclidean Distance
                dist = torch.norm(z_full_batch - z_partial, dim=1).sum().item()
                shift_dist += dist
                count += batch_size
                
        # Average shift per sample
        return shift_dist / count

    def benchmark(self, suffix=""):
        results = []
        
        print(f"\n=== Starting Benchmark {suffix} ===")
        
        for name in self.exp_dirs.keys():
            try:
                model = self.load_model(name)
                
                # 1. Latent Quality
                z = self.get_latents(model)
                acc, nmi = self.run_clustering_test(z)
                
                # 2. Robustness
                z_shift = self.run_robustness_test(model, z)
                
                results.append({
                    "Model": name,
                    "NMI": nmi,
                    "ACC": acc,
                    "Latent Shift": z_shift
                })
                print(f"✅ {name}: NMI={nmi:.4f}, ACC={acc:.4f}, Shift={z_shift}")
                
            except Exception as e:
                print(f"❌ Failed to eval {name}: {e}")
                # import traceback
                # traceback.print_exc()

        # Save CSV
        df = pd.DataFrame(results)
        csv_name = f"caltech_benchmark_{suffix}.csv"
        df.to_csv(csv_name, index=False)
        print(f"Saved results to {csv_name}")
        
        # Plot
        self.plot_results(df, suffix)

    def plot_results(self, df, suffix=""):
        # Filter out NaN shift (SMLVM) for the line plot
        df_line = df.dropna(subset=["Latent Shift"])
        
        fig, ax1 = plt.subplots(figsize=(14, 7))
        
        # Single Bar Plot: NMI
        # Using a distinct palette
        sns.barplot(data=df, x="Model", y="NMI", ax=ax1, palette="viridis", alpha=0.7)
        ax1.set_ylabel("Clustering NMI (Higher is Better)", fontsize=14)
        ax1.set_ylim(0, 1.0)
        ax1.tick_params(axis='x', rotation=45)
        
        # Line Plot: Latent Shift (Secondary Axis)
        if not df_line.empty:
            ax2 = ax1.twinx()
            sns.lineplot(data=df_line, x="Model", y="Latent Shift", ax=ax2, 
                         marker='o', color='red', linewidth=3, sort=False, label='Latent Shift')
            ax2.set_ylabel("Latent Shift (Lower is Better)", color='red', fontsize=14)
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.grid(False) # avoid clutter
            
        plt.title(f"Algorithm Performance Gap: Semantic Quality vs. Robustness ({suffix})", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"caltech_benchmark_gap_{suffix}.png")
        print(f"Saved plot: caltech_benchmark_gap_{suffix}.png")

if __name__ == "__main__":
    # Define Groups
    # We want to compare Z=10 specifically as usually high dim is more interesting
    # But let's run both if directories exist
    
    # Check what exists
    # Assuming config names match what we generated
    
    z10_experiments = {
        "SMLVM (Uncoded)": "caltech_smlvm_Z10_L1",
        "VAE-MLP (Uncoded)": "caltech_vae_mlp_Z10_L1_uncoded",
        "VAE-MLP (Rep L=2)": "caltech_vae_mlp_Z10_L2_rep",
        "VAE-MLP (Rand L=5)": "caltech_vae_mlp_Z10_L5_random",
        "VAE-CNN (Uncoded)": "caltech_vae_cnn_Z10_L1_uncoded",
        "VAE-CNN (Rep L=2)": "caltech_vae_cnn_Z10_L2_rep",
        "VAE-CNN (Rand L=5)": "caltech_vae_cnn_Z10_L5_random",
    }
    
    z5_experiments = {
        "SMLVM (Uncoded)": "caltech_smlvm_Z5_L1",
        "VAE-MLP (Uncoded)": "caltech_vae_mlp_Z5_L1_uncoded",
        "VAE-MLP (Rep L=2)": "caltech_vae_mlp_Z5_L2_rep",
        "VAE-MLP (Rand L=5)": "caltech_vae_mlp_Z5_L5_random",
        "VAE-CNN (Uncoded)": "caltech_vae_cnn_Z5_L1_uncoded",
        "VAE-CNN (Rep L=2)": "caltech_vae_cnn_Z5_L2_rep",
        "VAE-CNN (Rand L=5)": "caltech_vae_cnn_Z5_L5_random",
    }
    
    print("\n>>> Running Benchmark for Z=10 Group")
    comparator_z10 = AdvancedComparator(z10_experiments, device='cpu')
    comparator_z10.benchmark(suffix="Z10")

    print("\n>>> Running Benchmark for Z=5 Group")
    comparator_z5 = AdvancedComparator(z5_experiments, device='cpu')
    comparator_z5.benchmark(suffix="Z5")
