#!/usr/bin/env python
"""Quick evaluation script - CPU only, minimal dependencies."""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import numpy as np
import glob

# Disable sklearn parallelism
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

from src.models.cng_model import CNG_MV_GPLVM
from src.utils.data_utils import get_dataset

# Load data
print("Loading dataset...")
dataset = get_dataset('100leaves')
labels = dataset.labels.numpy()
view_dims = {k: v.shape[1] for k, v in dataset.views.items()}
num_classes = len(np.unique(labels))
print(f'Loaded 100Leaves: N={len(labels)}, Classes={num_classes}')

def eval_model(name, ckpt_path, inf_mode, use_ecc, ecc_mode='repetition', redundancy=1):
    print(f"\nEvaluating {name}...")
    model = CNG_MV_GPLVM(
        num_data=1600, input_dim=32, view_dims=view_dims,
        inference_mode=inf_mode, use_ecc=use_ecc, ecc_mode=ecc_mode,
        redundancy_factor=redundancy, rff_samples=1000
    )
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
    model.eval()
    
    if inf_mode == 'direct':
        z = model.q_mu.detach().numpy()
    else:
        views = dataset.views
        with torch.no_grad():
            if inf_mode == 'semi_amortized':
                _, (mu, _) = model.get_latents(views_batch=views)
            else:
                mu, _ = model.encoder(views)
            z = mu.numpy()
    
    print(f"  Latent shape: {z.shape}, mean: {z.mean():.4f}, std: {z.std():.4f}")
    
    # Use simple K-Means from scratch to avoid library issues
    from scipy.spatial.distance import cdist
    
    # Initialize centroids with k-means++
    np.random.seed(42)
    centroids = [z[np.random.randint(len(z))]]
    for _ in range(num_classes - 1):
        dists = cdist(z, np.array(centroids)).min(axis=1)
        probs = dists ** 2 / (dists ** 2).sum()
        centroids.append(z[np.random.choice(len(z), p=probs)])
    centroids = np.array(centroids)
    
    # Run K-Means iterations
    for _ in range(50):
        dists = cdist(z, centroids)
        y_pred = dists.argmin(axis=1)
        for k in range(num_classes):
            if (y_pred == k).sum() > 0:
                centroids[k] = z[y_pred == k].mean(axis=0)
    
    # Compute NMI
    from collections import Counter
    
    # Simplified NMI calculation
    def mutual_info(y_true, y_pred):
        contingency = {}
        for t, p in zip(y_true, y_pred):
            contingency[(t, p)] = contingency.get((t, p), 0) + 1
        
        n = len(y_true)
        mi = 0.0
        true_counts = Counter(y_true)
        pred_counts = Counter(y_pred)
        
        for (t, p), count in contingency.items():
            pxy = count / n
            px = true_counts[t] / n
            py = pred_counts[p] / n
            if pxy > 0:
                mi += pxy * np.log(pxy / (px * py))
        
        # Entropy
        h_true = -sum((c/n) * np.log(c/n) for c in true_counts.values() if c > 0)
        h_pred = -sum((c/n) * np.log(c/n) for c in pred_counts.values() if c > 0)
        
        nmi = 2 * mi / (h_true + h_pred) if (h_true + h_pred) > 0 else 0
        return nmi
    
    nmi = mutual_info(labels.astype(int), y_pred)
    
    # Cluster accuracy via Hungarian
    def cluster_acc(y_true, y_pred):
        from scipy.optimize import linear_sum_assignment
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(len(y_pred)):
            w[y_pred[i], y_true[i]] += 1
        row_ind, col_ind = linear_sum_assignment(w.max() - w)
        return w[row_ind, col_ind].sum() / len(y_pred)
    
    acc = cluster_acc(labels.astype(int), y_pred)
    
    print(f'  {name}: NMI={nmi:.4f}, ACC={acc:.4f}')
    return nmi, acc

if __name__ == "__main__":
    print('\n=== Evaluation Results (Z=32, L=5) ===')
    
    results = []
    
    # Find checkpoints by pattern
    ckpts = {}
    for pattern, name in [
        ('yang_direct_Z32_L5', 'Yang-Direct'),
        ('yang_amortized_mlp_Z32_L5', 'Yang-Amort-MLP'),
        ('semi_mlp_Z32_L5_rep', 'Semi-MLP-Rep'),
        ('semi_mlp_Z32_L5_random', 'Semi-MLP-Rand'),
    ]:
        matches = glob.glob(f'checkpoints/100leaves/{pattern}*')
        if matches:
            ckpts[name] = matches[0]
    
    configs = {
        'Yang-Direct': ('direct', False, 'repetition', 1),
        'Yang-Amort-MLP': ('amortized', False, 'repetition', 1),
        'Semi-MLP-Rep': ('semi_amortized', True, 'repetition', 5),
        'Semi-MLP-Rand': ('semi_amortized', True, 'random_gaussian', 5),
    }
    
    for name, ckpt_dir in ckpts.items():
        ckpt_path = os.path.join(ckpt_dir, 'final_model.pth')
        inf_mode, use_ecc, ecc_mode, redundancy = configs[name]
        try:
            nmi, acc = eval_model(name, ckpt_path, inf_mode, use_ecc, ecc_mode, redundancy)
            results.append({'Model': name, 'NMI': nmi, 'ACC': acc})
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
    
    # Print summary
    print("\n" + "="*50)
    print("Summary Table")
    print("="*50)
    for r in results:
        print(f"{r['Model']:20s}: NMI={r['NMI']:.4f}, ACC={r['ACC']:.4f}")
