
import os
import torch
import numpy as np
import yaml
import argparse
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Local Imports
from src.models.cng_model import CNG_MV_GPLVM
from src.utils.data_utils import get_dataset

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size

class UnifiedAnalyzer:
    def __init__(self, device='cpu'):
        self.device = device
        self.dataset_cache = {}

    def get_cached_dataset(self, dataset_name):
        if dataset_name not in self.dataset_cache:
            print(f"[Analyzer] Loading dataset: {dataset_name}...")
            self.dataset_cache[dataset_name] = get_dataset(dataset_name)
        return self.dataset_cache[dataset_name]

    def analyze_folder(self, exp_dir, do_tsne=False):
        print(f"\n--- Analyzing: {os.path.basename(exp_dir)} ---")
        
        # 1. Load Config
        config_path = os.path.join(exp_dir, "config.yaml")
        if not os.path.exists(config_path):
            print(f"Error: No config.yaml in {exp_dir}")
            return None
            
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            
        dataset_name = cfg['experiment']['dataset']
        dataset = self.get_cached_dataset(dataset_name)
        loader = DataLoader(dataset, batch_size=512, shuffle=False)
        
        # 2. Reconstruct Model
        view_dims = {k: v.shape[1] for k, v in dataset.views.items()}
        num_data = len(dataset)
        
        model = CNG_MV_GPLVM(
            num_data=num_data,
            input_dim=cfg['latent_space']['info_dim'],
            view_dims=view_dims,
            redundancy_factor=cfg['latent_space']['redundancy_factor'],
            use_ecc=(cfg['latent_space']['ecc_type'] != 'none'),
            ecc_mode=cfg['latent_space']['ecc_type'],
            inference_mode=cfg['model']['inference_mode'],
            encoder_type=cfg['model']['encoder_type'],
            num_mixtures=cfg['kernels']['num_mixtures'],
            rff_samples=cfg['kernels']['rff_samples']
        ).to(self.device)
        
        # 3. Load State Dict
        ckpt_path = os.path.join(exp_dir, "checkpoints", "final_model.pth")
        if not os.path.exists(ckpt_path):
            print(f"Error: No checkpoint at {ckpt_path}")
            return None
            
        model.load_state_dict(torch.load(ckpt_path, map_location=self.device), strict=False)
        model.eval()
        
        # 4. Extract Latents
        z_list = []
        with torch.no_grad():
            if model.inference_mode == 'direct':
                z_data = model.q_mu.detach().cpu().numpy()
            else:
                for views, _, _ in loader:
                    views = {k: v.to(self.device).float() for k, v in views.items()}
                    if model.inference_mode == 'semi_amortized':
                        _, (mu_enc, _) = model.get_latents(views_batch=views)
                        z_list.append(mu_enc.cpu())
                    else:
                        mu, _ = model.encoder(views)
                        z_list.append(mu.cpu())
                z_data = torch.cat(z_list, dim=0).numpy()
        
        # 5. Metrics
        labels = dataset.labels.numpy()
        num_classes = len(np.unique(labels))
        
        kmeans = KMeans(n_clusters=num_classes, n_init=20, random_state=42)
        y_pred = kmeans.fit_predict(z_data)
        
        acc = cluster_acc(labels, y_pred)
        nmi = normalized_mutual_info_score(labels, y_pred)
        
        print(f"Results: NMI={nmi:.4f}, ACC={acc:.4f}")
        
        # 6. Optional T-SNE
        if do_tsne:
            print("[Analyzer] Running t-SNE...")
            tsne = TSNE(n_components=2, random_state=42)
            z_embedded = tsne.fit_transform(z_data)
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(z_embedded[:, 0], z_embedded[:, 1], c=labels, cmap='tab20', s=10, alpha=0.6)
            plt.title(f"t-SNE: {cfg['experiment']['name']}\nNMI: {nmi:.3f}")
            plt.colorbar(scatter)
            save_path = os.path.join(exp_dir, "latent_tsne.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved t-SNE to {save_path}")
            
        return {"name": cfg['experiment']['name'], "nmi": nmi, "acc": acc}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Experiment Analyzer")
    parser.add_argument("folders", nargs="+", help="Path to experiment folders (one or more)")
    parser.add_argument("--tsne", action="store_true", help="Generate t-SNE plot")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    
    # Device selection
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    else:
        device = torch.device(args.device)
        
    analyzer = UnifiedAnalyzer(device=device)
    
    results = []
    for folder in args.folders:
        # Handle globbing if passed via shell
        if os.path.isdir(folder):
            res = analyzer.analyze_folder(folder, do_tsne=args.tsne)
            if res:
                results.append(res)
                
    if results:
        print("\n" + "="*40)
        print(f"{'Experiment Name':<30} | {'NMI':<6} | {'ACC':<6}")
        print("-"*40)
        for r in results:
            print(f"{r['name']:<30} | {r['nmi']:.4f} | {r['acc']:.4f}")
        print("="*40)
