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
from src.utils.data_utils import get_dataset

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

class SemiMfeatComparator:
    def __init__(self, checkpoint_dir, results_dir, device='cpu'):
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"Loading Mfeat Data... [Checkpoints: {checkpoint_dir}]")
        self.dataset = get_dataset(dataset_name="mfeat")
        self.loader = DataLoader(self.dataset, batch_size=512, shuffle=False)
        self.num_data = len(self.dataset)
        self.view_dims = {k: v.shape[1] for k, v in self.dataset.views.items()}
        self.labels = self.dataset.labels.numpy()
        self.num_classes = len(np.unique(self.labels))

    def load_model(self, exp_prefix, z, L):
        # Look directly in the provided checkpoint_dir
        pattern = os.path.join(self.checkpoint_dir, f"{exp_prefix}*")
        candidates = sorted(glob.glob(pattern))
        if not candidates:
            # Try recursive search if not found in root (engine creates subfolders)
            # Actually engine creates checkpoint_dir/{exp_name}_{timestamp}
            # So pattern above matches that directory.
            pass

        if not candidates:
            raise ValueError(f"No checkpoint found for prefix {exp_prefix} in {self.checkpoint_dir}")
        
        actual_dir = candidates[-1]
        name_lower = exp_prefix.lower()
        
        # Determine redundancy
        if "yang" in name_lower or "uncoded" in name_lower:
            redundancy = 1
        else:
            redundancy = L
            
        # Determine ECC mode
        ecc_mode = 'random_gaussian' if "random" in name_lower else 'repetition'
        use_ecc = redundancy > 1
        
        # Determine Inference Mode
        if "semi" in name_lower:
            inference_mode = 'semi_amortized'
        elif "yang_direct" in name_lower:
            inference_mode = 'direct'
        else:
            inference_mode = 'amortized'
            
        encoder_type = 'mlp'
            
        model = CNG_MV_GPLVM(
            num_data=self.num_data,
            input_dim=z,
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
        
        # Drop one view (e.g., 'mor' - smallest dimension)
        drop_views = ['mor'] 
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

    def run_benchmark(self, z, L):
        print(f"\n>>> Benchmarking Mfeat Z={z}, L={L}")
        models_map = {
            "Yang-Direct": f"yang_direct_Z{z}_L{L}",
            "Yang-Amort-MLP": f"yang_amortized_mlp_Z{z}_L{L}",
            "VAE-MLP-Uncoded": f"vae_mlp_Z{z}_L{L}_uncoded",
            "VAE-MLP-Rep": f"vae_mlp_Z{z}_L{L}_rep",
            "VAE-MLP-Rand": f"vae_mlp_Z{z}_L{L}_random",
            "Semi-MLP-Rep": f"semi_mlp_Z{z}_L{L}_rep",
            "Semi-MLP-Rand": f"semi_mlp_Z{z}_L{L}_random",
        }
        
        results = []
        for label, prefix in models_map.items():
            try:
                model = self.load_model(prefix, z, L)
                z_data = self.get_latents(model)
                
                kmeans = KMeans(n_clusters=self.num_classes, n_init=20, random_state=42)
                y_pred = kmeans.fit_predict(z_data)
                acc = cluster_acc(self.labels, y_pred)
                nmi = normalized_mutual_info_score(self.labels, y_pred)
                shift = self.run_robustness_test(model, z_data)
                
                results.append({"Model": label, "NMI": nmi, "ACC": acc, "Latent Shift": shift})
                print(f"✅ {label}: NMI={nmi:.4f}, ACC={acc:.4f}, Shift={shift:.4f}")
            except Exception as e:
                print(f"❌ Failed {label}: {e}")

        if not results:
             print("No results to plot.")
             return

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.results_dir, f"semi_mfeat_Z{z}_L{L}.csv"), index=False)
        self.plot(df, z, L)

    def plot(self, df, z, L):
        fig, ax1 = plt.subplots(figsize=(14, 7))
        sns.barplot(data=df, x="Model", y="NMI", ax=ax1, palette="viridis", alpha=0.7)
        ax1.set_ylabel("NMI", fontsize=14)
        ax1.set_ylim(0, 1.0)
        ax1.tick_params(axis='x', rotation=45)
        
        ax2 = ax1.twinx()
        sns.lineplot(data=df, x="Model", y="Latent Shift", ax=ax2, marker='o', color='red', linewidth=3, sort=False)
        ax2.set_ylabel("Latent Shift (Lower is Better)", color='red', fontsize=14)
        
        plt.title(f"Mfeat Semi-Amortized Benchmark (Z={z}, L={L})")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"semi_mfeat_Z{z}_L{L}.png"))
        plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/mfeat", help="Path to checkpoints")
    parser.add_argument("--results_dir", type=str, default="results/semi_mfeat", help="Path to save results")
    args = parser.parse_args()
    
    comp = SemiMfeatComparator(checkpoint_dir=args.checkpoint_dir, results_dir=args.results_dir)
    for z in [10, 20]:
        for L in [2, 5, 10]:
            comp.run_benchmark(z, L)
