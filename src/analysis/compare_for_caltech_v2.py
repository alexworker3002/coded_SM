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
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size

class CaltechV2Comparator:
    def __init__(self, exp_dirs, device='cpu', results_dir="results/caltech_v2"):
        self.device = device
        self.exp_dirs = exp_dirs
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
        print("Loading Caltech101-7 Data...")
        self.dataset = load_caltech_data()
        self.loader = DataLoader(self.dataset, batch_size=512, shuffle=False)
        self.num_data = len(self.dataset)
        self.view_dims = {k: v.shape[1] for k, v in self.dataset.views.items()}
        self.labels = self.dataset.labels.numpy()
        self.num_classes = len(np.unique(self.labels))

    def load_model(self, exp_prefix):
        candidates = glob.glob(f"checkpoints/{exp_prefix}*")
        if not candidates:
            raise ValueError(f"No checkpoint found for prefix {exp_prefix}")
        
        actual_dir = sorted(candidates)[-1]
        print(f"[{exp_prefix}] Loading from: {actual_dir}")

        name_lower = exp_prefix.lower()
        
        # Dimensions
        for z in [80, 40, 20, 10]:
            if f"z{z}" in name_lower:
                input_dim = z
                break
        else: input_dim = 10
            
        if "l5" in name_lower: redundancy = 5
        elif "l2" in name_lower: redundancy = 2
        else: redundancy = 1
            
        ecc_mode = 'random_gaussian' if "random" in name_lower else 'repetition'
        use_ecc = False if ("uncoded" in name_lower or redundancy == 1 or "smlvm" in name_lower) else True
        
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
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict, strict=False)
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

    def run_benchmark(self, z):
        experiments = {
            "SMLVM": f"caltech_v2_smlvm_Z{z}_L1",
            "VAE-MLP (Uncoded)": f"caltech_v2_vae_mlp_Z{z}_L1_uncoded",
            "VAE-MLP (Rep L=2)": f"caltech_v2_vae_mlp_Z{z}_L2_rep",
            "VAE-MLP (Rand L=5)": f"caltech_v2_vae_mlp_Z{z}_L5_random",
            "VAE-CNN (Uncoded)": f"caltech_v2_vae_cnn_Z{z}_L1_uncoded",
            "VAE-CNN (Rep L=2)": f"caltech_v2_vae_cnn_Z{z}_L2_rep",
            "VAE-CNN (Rand L=5)": f"caltech_v2_vae_cnn_Z{z}_L5_random",
        }
        
        results = []
        for label, prefix in experiments.items():
            try:
                model = self.load_model(prefix)
                z_data = self.get_latents(model)
                
                # Clustering
                kmeans = KMeans(n_clusters=self.num_classes, n_init=20, random_state=42)
                y_pred = kmeans.fit_predict(z_data)
                acc = cluster_acc(self.labels, y_pred)
                nmi = normalized_mutual_info_score(self.labels, y_pred)
                
                # Robustness (Mask half views)
                drop_views = ['gabor', 'wm', 'centrist']
                shift = 0
                if model.inference_mode == 'amortized':
                    count = 0
                    with torch.no_grad():
                        for views, _, _ in self.loader:
                            views = {k: v.to(self.device).float() for k, v in views.items()}
                            batch_size = list(views.values())[0].shape[0]
                            masked = {k: (v if k not in drop_views else torch.zeros_like(v)) for k, v in views.items()}
                            z_partial, _ = model.encoder(masked)
                            
                            z_full_batch = torch.tensor(z_data[count:count+batch_size]).to(self.device)
                            shift += torch.norm(z_full_batch - z_partial, dim=1).sum().item()
                            count += batch_size
                    shift /= count
                else:
                    shift = np.nan
                
                results.append({"Model": label, "NMI": nmi, "ACC": acc, "Latent Shift": shift})
                print(f"✅ {label}: NMI={nmi:.4f}, Shift={shift}")
            except Exception as e:
                print(f"❌ Failed {label}: {e}")

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.results_dir, f"benchmark_Z{z}.csv"), index=False)
        self.plot(df, z)

    def plot(self, df, z):
        fig, ax1 = plt.subplots(figsize=(14, 7))
        sns.barplot(data=df, x="Model", y="NMI", ax=ax1, palette="viridis", alpha=0.7)
        ax1.set_ylabel("NMI", fontsize=14)
        ax1.set_ylim(0, 1.0)
        ax1.tick_params(axis='x', rotation=45)
        
        ax2 = ax1.twinx()
        sns.lineplot(data=df, x="Model", y="Latent Shift", ax=ax2, marker='o', color='red', linewidth=3, sort=False)
        ax2.set_ylabel("Latent Shift (Lower is Better)", color='red', fontsize=14)
        
        plt.title(f"Caltech101-7 v2 Performance (Z={z})")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"benchmark_Z{z}.png"))
        plt.close()

if __name__ == "__main__":
    comp = CaltechV2Comparator({})
    for z in [10, 20, 40, 80]:
        print(f"\n>>> Analyzing Z={z}")
        comp.run_benchmark(z)
