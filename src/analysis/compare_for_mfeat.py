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
from src.utils.data_utils import load_mfeat_data

# Set Style
sns.set(style="whitegrid", context="talk")

def cluster_acc(y_true, y_pred):
    """Compute Clustering Accuracy"""
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size

class MfeatComparator:
    def __init__(self, exp_dirs, device='cpu'):
        self.device = device
        self.exp_dirs = exp_dirs
        
        print("Loading mfeat Data...")
        self.dataset = load_mfeat_data(mode="real")
        self.loader = DataLoader(self.dataset, batch_size=512, shuffle=False)
        self.num_data = len(self.dataset)
        self.view_dims = {k: v.shape[1] for k, v in self.dataset.views.items()}
        self.labels = self.dataset.labels.numpy()
        self.num_classes = len(np.unique(self.labels))
        print(f"Data Loaded: N={self.num_data}, Classes={self.num_classes}")

    def load_model(self, exp_name):
        ckpt_prefix = self.exp_dirs[exp_name]
        candidates = glob.glob(f"checkpoints/{ckpt_prefix}*")
        if not candidates:
            # Maybe path relative
            candidates = glob.glob(f"{ckpt_prefix}*")
            if not candidates:
                 # Check if user passed full path
                 if os.path.exists(ckpt_prefix):
                    candidates = [ckpt_prefix]
                 else:
                    raise ValueError(f"No checkpoint found for prefix {ckpt_prefix}")
        
        actual_dir = sorted(candidates)[-1]
        print(f"[{exp_name}] Loading from: {actual_dir}")

        name_lower = ckpt_prefix.lower()
        
        # Dimensions
        if "z30" in name_lower: input_dim = 30
        elif "z20" in name_lower: input_dim = 20
        elif "z15" in name_lower: input_dim = 15
        elif "z10" in name_lower: input_dim = 10
        else: input_dim = 10
            
        # Redundancy
        if "l5" in name_lower: redundancy = 5
        elif "l2" in name_lower: redundancy = 2
        else: redundancy = 1
            
        # ECC Mode
        if "random" in name_lower: ecc_mode = 'random_gaussian'
        elif "rep" in name_lower: ecc_mode = 'repetition'
        else: ecc_mode = 'repetition'
            
        use_ecc = True
        if "uncoded" in name_lower or "l1" in name_lower or "smlvm" in name_lower:
            if redundancy == 1:
                use_ecc = False
        
        if "vae" in name_lower:
            inference_mode = 'amortized'
            encoder_type = 'cnn' if 'cnn' in name_lower else 'mlp'
        else:
            inference_mode = 'direct'
            encoder_type = 'mlp'
            
        model = CNG_MV_GPLVM(
            num_data=self.num_data,
            input_dim=input_dim,
            view_dims=self.view_dims,
            redundancy_factor=redundancy,
            use_ecc=use_ecc,
            ecc_mode=ecc_mode,
            inference_mode=inference_mode,
            encoder_type=encoder_type,
            rff_samples=1000
        ).to(self.device)
        
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
        if model.inference_mode == 'direct':
            return model.q_mu.detach().cpu().numpy()
            
        z_list = []
        with torch.no_grad():
            for views, _, _ in self.loader:
                views = {k: v.to(self.device).float() for k, v in views.items()}
                mu, _ = model.encoder(views)
                z_list.append(mu.cpu())
        return torch.cat(z_list, dim=0).numpy()

    def run_clustering_test(self, z_data):
        kmeans = KMeans(n_clusters=self.num_classes, n_init=20, random_state=42)
        y_pred = kmeans.fit_predict(z_data)
        acc = cluster_acc(self.labels, y_pred)
        nmi = normalized_mutual_info_score(self.labels, y_pred)
        return acc, nmi

    def run_robustness_test(self, model, full_z):
        """
        Mask 'fac' (216 dim) and 'pix' (240 dim) - the two largest/richest views for mfeat.
        """
        if model.inference_mode == 'direct':
            return np.nan 
            
        shift_dist = 0
        count = 0
        drop_views = ['fac', 'pix']
        
        with torch.no_grad():
            for i, (views, _, _) in enumerate(self.loader):
                views = {k: v.to(self.device).float() for k, v in views.items()}
                batch_size = list(views.values())[0].shape[0]
                
                masked_views = {k: (v if k not in drop_views else torch.zeros_like(v)) 
                                for k, v in views.items()}
                
                model.eval() # important for batchnorm
                z_partial, _ = model.encoder(masked_views)
                
                start_idx = i * self.loader.batch_size
                z_full_batch = torch.tensor(full_z[start_idx : start_idx+batch_size]).to(self.device)
                
                dist = torch.norm(z_full_batch - z_partial, dim=1).sum().item()
                shift_dist += dist
                count += batch_size
                
        return shift_dist / count

    def benchmark(self, suffix=""):
        results = []
        print(f"\n=== BENCHMARK {suffix} ===")
        for name in self.exp_dirs.keys():
            try:
                model = self.load_model(name)
                z = self.get_latents(model)
                acc, nmi = self.run_clustering_test(z)
                z_shift = self.run_robustness_test(model, z)
                
                results.append({
                    "Model": name,
                    "NMI": nmi, "ACC": acc, "Latent Shift": z_shift
                })
                print(f"✅ {name}: NMI={nmi:.4f}, Shift={z_shift}")
            except Exception as e:
                print(f"❌ Failed {name}: {e}")

        df = pd.DataFrame(results)
        df.to_csv(f"mfeat_benchmark_{suffix}.csv", index=False)
        self.plot_results(df, suffix)

    def plot_results(self, df, suffix=""):
        fig, ax1 = plt.subplots(figsize=(14, 7))
        sns.barplot(data=df, x="Model", y="NMI", ax=ax1, palette="viridis", alpha=0.7)
        ax1.set_ylabel("NMI (Higher is Better)", fontsize=14)
        ax1.set_ylim(0, 1.0)
        ax1.tick_params(axis='x', rotation=45)
        
        ax2 = ax1.twinx()
        sns.lineplot(data=df, x="Model", y="Latent Shift", ax=ax2, 
                     marker='o', color='red', linewidth=3, sort=False, label='Latent Shift')
        ax2.set_ylabel("Latent Shift (Lower is Better)", color='red', fontsize=14)
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.grid(False)
        
        plt.title(f"Mfeat Performance Gap ({suffix})")
        plt.tight_layout()
        plt.savefig(f"mfeat_benchmark_gap_{suffix}.png")
        print(f"Saved plot: mfeat_benchmark_gap_{suffix}.png")

if __name__ == "__main__":
    dims = [10, 15, 20, 30]
    
    for z in dims:
        experiments = {
            "SMLVM": f"mfeat_smlvm_Z{z}_L1",
            "VAE-MLP (Uncoded)": f"mfeat_vae_mlp_Z{z}_L1_uncoded",
            "VAE-MLP (Rep L=2)": f"mfeat_vae_mlp_Z{z}_L2_rep",
            "VAE-MLP (Rand L=5)": f"mfeat_vae_mlp_Z{z}_L5_random",
            "VAE-CNN (Uncoded)": f"mfeat_vae_cnn_Z{z}_L1_uncoded",
            "VAE-CNN (Rep L=2)": f"mfeat_vae_cnn_Z{z}_L2_rep",
            "VAE-CNN (Rand L=5)": f"mfeat_vae_cnn_Z{z}_L5_random",
        }
        
        comp = MfeatComparator(experiments, device='cpu')
        comp.benchmark(suffix=f"Z{z}")
