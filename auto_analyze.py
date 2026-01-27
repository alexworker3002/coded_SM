#!/usr/bin/env python
"""
Auto-Analyze: Automated experiment analysis with visualization
Generates comprehensive metrics + comparison plots
"""

import os
import sys
import glob
import torch
import numpy as np
import yaml
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from src.models.cng_model import CNG_MV_GPLVM
from src.utils.data_utils import get_dataset

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size

def analyze_experiment(exp_dir, device='cuda'):
    """Comprehensive analysis of a single experiment"""
    
    config_path = os.path.join(exp_dir, "config.yaml")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    exp_name = cfg['experiment']['name']
    print(f"  Processing: {exp_name}")
    
    dataset_name = cfg['experiment']['dataset']
    dataset = get_dataset(dataset_name)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    
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
    ).to(device)
    
    ckpt_path = os.path.join(exp_dir, "final_model.pth")
    if not os.path.exists(ckpt_path):
        # Fallback for older structure
        ckpt_path_alt = os.path.join(exp_dir, "checkpoints", "final_model.pth")
        if os.path.exists(ckpt_path_alt):
            ckpt_path = ckpt_path_alt
        else:
            print(f"  ⚠️  No checkpoint found: {exp_name}")
            return None
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False), strict=False)
    model.eval()
    
    z_opt_list = []
    z_enc_list = []
    recon_errors_per_view = {name: [] for name in view_dims.keys()}
    
    with torch.no_grad():
        for views, _, indices in loader:
            views = {k: v.to(device).float() for k, v in views.items()}
            indices = indices.to(device)
            
            if model.inference_mode == 'semi_amortized':
                (mu_opt, _), (mu_enc, _) = model.get_latents(indices, views)
                z_opt_list.append(mu_opt.cpu())
                z_enc_list.append(mu_enc.cpu())
                
                outputs = model(indices, views)
                y_recons = outputs[0]
                
                for view_name in view_dims.keys():
                    y_true = views[view_name]
                    y_pred = y_recons[view_name]
                    error = torch.mean((y_true - y_pred)**2, dim=1)
                    recon_errors_per_view[view_name].append(error.cpu())
            
            elif model.inference_mode == 'direct':
                mu_opt = model.q_mu[indices]
                z_opt_list.append(mu_opt.cpu())
                z_enc_list.append(mu_opt.cpu())
            else:
                mu, _ = model.get_latents(views_batch=views)
                z_enc_list.append(mu.cpu())
                z_opt_list.append(mu.cpu())
    
    z_opt = torch.cat(z_opt_list, dim=0).numpy()
    z_enc = torch.cat(z_enc_list, dim=0).numpy()
    
    labels = dataset.labels.numpy()
    num_classes = len(np.unique(labels))
    
    kmeans = KMeans(n_clusters=num_classes, n_init=20, random_state=42)
    y_pred = kmeans.fit_predict(z_enc)
    
    nmi = normalized_mutual_info_score(labels, y_pred)
    acc = cluster_acc(labels, y_pred)
    
    if model.inference_mode == 'semi_amortized':
        latent_shift = np.mean(np.linalg.norm(z_opt - z_enc, axis=1))
        latent_shift_std = np.std(np.linalg.norm(z_opt - z_enc, axis=1))
    else:
        latent_shift = 0.0
        latent_shift_std = 0.0
    
    if recon_errors_per_view[list(view_dims.keys())[0]]:
        recon_dict = {}
        for view_name, errors in recon_errors_per_view.items():
            errors_tensor = torch.cat(errors)
            recon_dict[f'recon_{view_name}'] = errors_tensor.mean().item()
        avg_recon = np.mean(list(recon_dict.values()))
    else:
        recon_dict = {}
        avg_recon = 0.0
    
    z_mean_norm = np.mean(np.linalg.norm(z_enc, axis=1))
    z_std = np.std(z_enc)
    
    results = {
        'experiment': exp_name,
        'latent_dim': cfg['latent_space']['info_dim'],
        'redundancy': cfg['latent_space']['redundancy_factor'],
        'ecc_type': cfg['latent_space']['ecc_type'],
        'encoder': cfg['model']['encoder_type'],
        'mode': cfg['model']['inference_mode'],
        'nmi': nmi,
        'acc': acc,
        'latent_shift_mean': latent_shift,
        'latent_shift_std': latent_shift_std,
        'avg_recon_error': avg_recon,
        'z_mean_norm': z_mean_norm,
        'z_std': z_std,
    }
    results.update(recon_dict)
    
    return results

def generate_visualizations(df, output_dir):
    """Generate professional comparative visualization plots"""
    
    print("\n[Viz] Generating professional comparison plots...")
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    # Pre-process labels for cleaner display
    def clean_label(row):
        if row['mode'] == 'amortized' and row['encoder'] == 'mlp':
            return "Yang_Base"
        arch = row['encoder'].upper()
        mode_str = "Semi" # Everything else in our current setup is semi
        ecc = row['ecc_type'][:4].capitalize() if row['ecc_type'] != 'none' else 'Base'
        L = f"_L{row['redundancy']}" if row['ecc_type'] != 'none' else ""
        return f"{arch}_{mode_str}_{ecc}{L}"
    
    df['display_name'] = df.apply(clean_label, axis=1)
    z_dims = sorted(df['latent_dim'].unique())
    
    # 1. Performance Matrix: ACC (Bars) + NMI (Line/Points)
    # ---------------------------------------------------
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Colors for different latent dimensions
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(z_dims)))
    
    x_labels = df[df['latent_dim'] == z_dims[0]]['display_name'].tolist()
    x = np.arange(len(x_labels))
    width = 0.8 / len(z_dims)
    
    for i, z in enumerate(z_dims):
        subset = df[df['latent_dim'] == z].copy()
        # Sort by display order to match x_labels
        subset['sort_idx'] = subset['display_name'].apply(lambda d: x_labels.index(d) if d in x_labels else 99)
        subset = subset.sort_values('sort_idx')
        
        offset = (i - (len(z_dims)-1)/2) * width
        
        # ACC as Bars
        ax1.bar(x + offset, subset['acc'], width=width, color=colors[i], alpha=0.6, label=f'ACC (Z={z})')
        # NMI as dots with line
        ax2.plot(x + offset, subset['nmi'], 'o-', color=colors[i], markersize=6, linewidth=1.5, label=f'NMI (Z={z})')

    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=45)
    ax1.set_ylabel('Accuracy (ACC)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('NMI Score', fontsize=12, fontweight='bold')
    ax1.set_title('Model Performance: Accuracy vs NMI', fontsize=14, fontweight='bold', pad=20)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(z_dims), frameon=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "performance_matrix.png"), bbox_inches='tight')
    plt.close()

    # 2. Stability Trade-off: ACC (Bars) vs Latent Shift (Line)
    # ---------------------------------------------------------
    for z in z_dims:
        subset = df[df['latent_dim'] == z].copy()
        subset = subset.sort_values(['redundancy', 'ecc_type']) # Logical order
        
        plt.figure(figsize=(10, 6))
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        x_names = subset['display_name'].tolist()
        x_pos = np.arange(len(x_names))
        
        # Primary: Accuracy
        bars = ax1.bar(x_pos, subset['acc'], color='skyblue', alpha=0.7, label='Accuracy')
        ax1.set_ylabel('Accuracy', fontsize=12, color='blue', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Secondary: Latent Shift
        line = ax2.plot(x_pos, subset['latent_shift_mean'], 'rD-', linewidth=2, markersize=8, label='Latent Shift')
        ax2.set_ylabel('Latent Shift (Stability Mirror)', fontsize=12, color='red', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Set shared x-axis
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(x_names, rotation=30)
        ax1.set_title(f'Stability vs Performance (Z={z})', fontsize=14, fontweight='bold')
        
        # Annotate actual values on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.2f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"stability_tradeoff_z{z}.png"))
        plt.close()

    # 3. Comprehensive Summary Table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    table_data = []
    headers = ['Model', 'Z', 'Arch', 'Redun', 'NMI', 'ACC', 'Latent Shift', 'Status']
    
    for _, row in df.iterrows():
        # Global baseline for this Z: Always Yang (Pure MLP)
        yang_baselines = df[(df['latent_dim'] == row['latent_dim']) & 
                            (df['mode'] == 'amortized') & 
                            (df['encoder'] == 'mlp')]
        
        if len(yang_baselines) > 0:
            baseline = yang_baselines.iloc[0]
            delta = row['acc'] - baseline['acc']
            status = "✅ Improve" if delta > 0.02 else ("💥 Collapse" if delta < -0.1 else "➖ Stable")
            if row['mode'] == 'amortized': status = "⭐ YANG"
            elif row['ecc_type'] == 'none': status = "⭐ Base"
        else:
            status = "N/A"
        
        table_data.append([
            row['display_name'], row['latent_dim'], row['encoder'].upper(), row['redundancy'],
            f"{row['nmi']:.4f}", f"{row['acc']:.4f}", f"{row['latent_shift_mean']:.4f}",
            status
        ])
    
    table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    plt.title('Comprehensive Experiment Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(os.path.join(fig_dir, "experiment_summary_table.png"), bbox_inches='tight')
    plt.close()
    
    print(f"[Viz] Professional visualizations saved to {fig_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=str, default="experiments/*", help="Experiment folder pattern")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"[Auto-Analyze] Using device: {device}")
    
    exp_folders = sorted(glob.glob(args.experiments))
    if not exp_folders:
        print(f"❌ No experiments found matching: {args.experiments}")
        sys.exit(1)
    
    print(f"[Auto-Analyze] Found {len(exp_folders)} experiments")
    
    all_results = []
    for exp_dir in exp_folders:
        try:
            res = analyze_experiment(exp_dir, device=device)
            if res:
                all_results.append(res)
        except Exception as e:
            print(f"  ❌ ERROR: {os.path.basename(exp_dir)}: {e}")
    
    if not all_results:
        print("❌ No successful analyses")
        sys.exit(1)
    
    df = pd.DataFrame(all_results)
    df = df.sort_values(['latent_dim', 'redundancy', 'ecc_type'])
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Save CSV
    csv_path = os.path.join(args.output, "comprehensive_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Metrics saved to: {csv_path}")
    
    # Generate visualizations
    generate_visualizations(df, args.output)
    
    # Print summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    summary_cols = ['experiment', 'latent_dim', 'redundancy', 'ecc_type', 'nmi', 'acc', 'latent_shift_mean']
    print(df[summary_cols].to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    main()
