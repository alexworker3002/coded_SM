import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import glob

# Local Imports
from src.models.cng_model import CNG_MV_GPLVM
from src.utils.data_utils import load_mfeat_data

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

class SemiComparator:
    def __init__(self, device='cpu', results_dir="results/semi_mfeat"):
        self.device = device
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
        print("Loading mfeat Data...")
        self.dataset = load_mfeat_data(mode="real")
        self.loader = DataLoader(self.dataset, batch_size=512, shuffle=False)
        self.num_data = len(self.dataset)
        self.view_dims = {k: v.shape[1] for k, v in self.dataset.views.items()}
        self.labels = self.dataset.labels.numpy()
        self.num_classes = len(np.unique(self.labels))

    def load_model(self, exp_prefix):
        candidates = sorted(glob.glob(f"checkpoints/{exp_prefix}*"))
        if not candidates:
            # try relative or direct path
            candidates = sorted(glob.glob(f"{exp_prefix}*"))
            if not candidates:
                 raise ValueError(f"No checkpoint found for prefix {exp_prefix}")
        
        actual_dir = candidates[-1]
        print(f"[{exp_prefix}] Loading from: {actual_dir}")
        name_lower = exp_prefix.lower()
        
        # Architecture detection from name
        input_dim = 10
        redundancy = 2 if "l2" in name_lower else 1
        
        if "random" in name_lower: ecc_mode = 'random_gaussian'
        else: ecc_mode = 'repetition'
        
        # ECC logic
        if "uncoded" in name_lower or "smlvm" in name_lower or redundancy == 1:
            use_ecc = False
        else:
            use_ecc = True
            
        # Inference mode
        if "semi" in name_lower and not "smlvm" in name_lower:
            inference_mode = 'semi_amortized'
        elif "vae" in name_lower:
            inference_mode = 'amortized'
        else:
            inference_mode = 'direct'
            
        encoder_type = 'cnn' if 'cnn' in name_lower else 'mlp'
            
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
                if model.inference_mode == 'semi_amortized':
                    _, (mu_enc, _) = model.get_latents(views_batch=views)
                    z_list.append(mu_enc.cpu())
                else:
                    mu = model.encoder(views)[0]
                    z_list.append(mu.cpu())
        return torch.cat(z_list, dim=0).numpy()

    def run_robustness_test(self, model, full_z):
        if model.inference_mode == 'direct':
            return np.nan 
        
        drop_views = ['fac', 'pix']
        shift = 0
        count = 0
        
        with torch.no_grad():
            for i, (views, _, _ ) in enumerate(self.loader):
                views = {k: v.to(self.device).float() for k, v in views.items()}
                batch_size = list(views.values())[0].shape[0]
                masked = {k: (v if k not in drop_views else torch.zeros_like(v)) for k, v in views.items()}
                
                if model.inference_mode == 'semi_amortized':
                    _, (mu_enc, _) = model.get_latents(views_batch=masked)
                    z_partial = mu_enc
                else:
                    z_partial, _ = model.encoder(masked)
                    
                start = i * self.loader.batch_size
                z_full = torch.tensor(full_z[start:start+batch_size]).to(self.device)
                shift += torch.norm(z_full - z_partial, dim=1).sum().item()
                count += batch_size
        return shift / count

    def run_benchmark(self):
        # 11 Models to compare
        experiments = {
            "SMLVM (Direct)": "semi_mfeat_smlvm_Z10",
            "VAE-MLP (Uncoded)": "semi_mfeat_vae_mlp_Z10_uncoded",
            "VAE-CNN (Uncoded)": "semi_mfeat_vae_cnn_Z10_uncoded",
            "VAE-MLP-Rep": "semi_mfeat_vae_mlp_Z10_L2_rep",
            "VAE-MLP-Rand": "semi_mfeat_vae_mlp_Z10_L2_random",
            "VAE-CNN-Rep": "semi_mfeat_vae_cnn_Z10_L2_rep",
            "VAE-CNN-Rand": "semi_mfeat_vae_cnn_Z10_L2_random",
            "Semi-MLP-Rep": "semi_mfeat_semi_mlp_Z10_L2_rep",
            "Semi-MLP-Rand": "semi_mfeat_semi_mlp_Z10_L2_random",
            "Semi-CNN-Rep": "semi_mfeat_semi_cnn_Z10_L2_rep",
            "Semi-CNN-Rand": "semi_mfeat_semi_cnn_Z10_L2_random",
        }
        
        results = []
        for label, prefix in experiments.items():
            try:
                model = self.load_model(prefix)
                z_data = self.get_latents(model)
                
                kmeans = KMeans(n_clusters=self.num_classes, n_init=20, random_state=42)
                y_pred = kmeans.fit_predict(z_data)
                acc = cluster_acc(self.labels, y_pred)
                nmi = normalized_mutual_info_score(self.labels, y_pred)
                shift = self.run_robustness_test(model, z_data)
                
                results.append({"Model": label, "NMI": nmi, "ACC": acc, "Latent Shift": shift})
                print(f"✅ {label}: NMI={nmi:.4f}, Shift={shift}")
            except Exception as e:
                print(f"❌ Failed {label}: {e}")

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.results_dir, "semi_benchmark_L2.csv"), index=False)
        self.plot(df)

    def plot(self, df):
        fig, ax1 = plt.subplots(figsize=(16, 8))
        # NMI Bar
        sns.barplot(data=df, x="Model", y="NMI", ax=ax1, palette="viridis", alpha=0.7)
        ax1.set_ylabel("NMI", fontsize=14)
        ax1.set_ylim(0, 1.0)
        ax1.tick_params(axis='x', rotation=45)
        
        # Latent Shift Line
        ax2 = ax1.twinx()
        sns.lineplot(data=df, x="Model", y="Latent Shift", ax=ax2, marker='o', color='red', linewidth=3, sort=False)
        ax2.set_ylabel("Latent Shift (Lower is Better)", color='red', fontsize=14)
        
        plt.title("Semi-Amortized L=2 Benchmark (Mfeat Z=10)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "semi_benchmark_L2.png"))
        print(f"Saved plot: {self.results_dir}/semi_benchmark_L2.png")

if __name__ == "__main__":
    comp = SemiComparator()
    comp.run_benchmark()
