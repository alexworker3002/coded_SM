#!/usr/bin/env python
"""Evaluate versioned checkpoints with explicit, reproducible metric definitions."""

import argparse
import copy
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader

from src.utils.data_utils import get_dataset
from src.utils.experiment import CHECKPOINT_VERSION, SEMI_MODES, build_model, config_digest
from src.utils.helpers import dataset_fingerprint, get_device, set_seed

METRIC_DEFINITIONS = {
    'evaluation_scope': 'Transductive clustering on the training dataset; no held-out generalization claim.',
    'latent_source': 'Encoder posterior mean for encoder modes; optimized posterior mean for direct.',
    'nmi': 'KMeans normalized mutual information, arithmetic averaging.',
    'acc': 'KMeans clustering accuracy after optimal Hungarian label matching.',
    'alignment_shift_mean': 'Mean L2 distance between optimized Z and encoder Z; semi modes only.',
    'robustness_shift_mean': 'Mean L2 shift when each view is omitted in turn, averaged over views and samples.',
    'robustness_relative_shift': 'robustness_shift_mean / mean norm of full-view encoder Z.',
    'masked_acc_mean': 'Mean clustering accuracy across leave-one-view-out encoder inputs.',
    'masked_nmi_mean': 'Mean NMI across leave-one-view-out encoder inputs.',
    'avg_recon_error': 'Mean per-view MSE at posterior means; only for trained reconstruction readouts.',
    'unavailable_values': 'NaN / blank CSV cells; never substituted with zero.',
    'uncertainty': 'Sample standard deviation across training seeds, not a confidence interval.',
}


def cluster_acc(y_true, y_pred):
    _, y_true = np.unique(y_true, return_inverse=True)
    _, y_pred = np.unique(y_pred, return_inverse=True)
    size = max(y_true.max(), y_pred.max()) + 1
    counts = np.zeros((size, size), dtype=np.int64)
    np.add.at(counts, (y_pred, y_true), 1)
    rows, columns = linear_sum_assignment(counts.max() - counts)
    return float(counts[rows, columns].sum() / len(y_true))


def clustering_metrics(z, labels, seed=42):
    classes = len(np.unique(labels))
    if not 1 < classes < len(labels):
        raise ValueError('Clustering evaluation requires repeated samples per class and at least two classes')
    if not np.isfinite(z).all():
        raise ValueError('Non-finite latent representation')
    with threadpool_limits(limits=1):
        pred = KMeans(n_clusters=classes, n_init=20, random_state=seed).fit_predict(z)
    return cluster_acc(labels, pred), float(normalized_mutual_info_score(labels, pred))


def analyze_experiment(exp_dir, device='cpu', evaluation_seed=42, robustness=True, dataset=None):
    exp_dir = Path(exp_dir)
    config_path = exp_dir / 'config.yaml'
    cfg = yaml.safe_load(config_path.read_text())
    ckpt_path = exp_dir / 'final_model.pth'
    if not ckpt_path.exists():
        ckpt_path = exp_dir / 'checkpoints/final_model.pth'
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != CHECKPOINT_VERSION:
        raise ValueError('Legacy/unversioned checkpoint: retrain after the ECC and loss fixes')
    if config_digest(cfg) != checkpoint['metadata']['config_sha256']:
        raise ValueError('Config does not match the checkpoint')
    if checkpoint['config'] != cfg:
        raise ValueError('Embedded checkpoint config does not match config.yaml')
    set_seed(evaluation_seed)
    if dataset is None:
        dataset = get_dataset(cfg['experiment']['dataset'], **cfg.get('dataset_kwargs', {}))
    if dataset.metadata.get('is_mock', False):
        raise ValueError('Benchmark evaluation refuses mock data')
    if dataset_fingerprint(dataset) != checkpoint['metadata']['data_fingerprint']:
        raise ValueError('Dataset values/order/preprocessing do not match training')
    model = build_model(cfg, dataset, device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    if model.inference_mode.startswith('coded_'):
        if not torch.equal(model.ecc_module.G, model.ecc_decoder.G):
            raise ValueError('Checkpoint contains inconsistent ECC matrices')
        if not torch.allclose(model.ecc_decoder.G_pinv,
                              torch.linalg.pinv(model.ecc_module.G), atol=1e-5, rtol=1e-4):
            raise ValueError('Checkpoint contains an invalid ECC inverse')
    model.eval()
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    mode = model.inference_mode
    z_enc_parts, z_opt_parts = [], []
    recon_errors = {name: [] for name in dataset.views}
    with torch.no_grad():
        for views, _, indices in loader:
            views = {key: value.to(device).float() for key, value in views.items()}
            indices = indices.to(device)
            if mode != 'direct':
                mu_enc, _ = model.infer_latents(views)
                z_enc_parts.append(mu_enc.cpu())
            if mode in ('direct', *SEMI_MODES):
                z_opt_parts.append(model.q_mu[indices].cpu())
            if not cfg['training'].get('use_gp_loss', True):
                reconstructions = model(indices, views)[0]
                for name, truth in views.items():
                    errors = (truth.reshape(len(indices), -1) - reconstructions[name]).square().mean(1)
                    recon_errors[name].append(errors.cpu())
    z_opt = torch.cat(z_opt_parts).numpy() if z_opt_parts else None
    z_enc = torch.cat(z_enc_parts).numpy() if z_enc_parts else None
    z = z_opt if mode == 'direct' else z_enc
    labels = dataset.labels.numpy()
    acc, nmi = clustering_metrics(z, labels, evaluation_seed)
    opt_acc, opt_nmi = (clustering_metrics(z_opt, labels, evaluation_seed)
                        if mode in SEMI_MODES else (np.nan, np.nan))
    alignment = np.linalg.norm(z_opt - z_enc, axis=1) if mode in SEMI_MODES else None
    missing_metrics, all_shifts = {}, []
    if robustness and z_enc is not None and len(dataset.views) > 1:
        with torch.no_grad():
            for omitted in dataset.views:
                parts = []
                for views, _, _ in loader:
                    available = {key: value.to(device).float() for key, value in views.items()
                                 if key != omitted}
                    mu, _ = model.infer_latents(available)
                    parts.append(mu.cpu())
                partial = torch.cat(parts).numpy()
                shifts = np.linalg.norm(z_enc - partial, axis=1)
                masked_acc, masked_nmi = clustering_metrics(partial, labels, evaluation_seed)
                missing_metrics[omitted] = dict(acc=masked_acc, nmi=masked_nmi,
                                                shift_mean=float(shifts.mean()))
                all_shifts.append(shifts)
    shift = float(np.concatenate(all_shifts).mean()) if all_shifts else np.nan
    recon = {f'recon_{name}': float(torch.cat(parts).mean())
             for name, parts in recon_errors.items() if parts}
    variant_cfg = copy.deepcopy(cfg)
    variant_cfg['experiment'].pop('seed', None)
    variant_cfg['experiment'].pop('name', None)
    try:
        checkpoint_display = str(ckpt_path.resolve().relative_to(Path.cwd()))
    except ValueError:
        checkpoint_display = str(ckpt_path.resolve())
    result = {
        'experiment': cfg['experiment']['name'], 'checkpoint': checkpoint_display,
        'dataset': cfg['experiment']['dataset'], 'seed': cfg['experiment'].get('seed', 42),
        'evaluation_seed': evaluation_seed, 'variant_id': config_digest(variant_cfg),
        'data_fingerprint': checkpoint['metadata']['data_fingerprint'],
        'source_sha256': checkpoint['metadata']['source_sha256'],
        'latent_dim': cfg['latent_space']['info_dim'],
        'redundancy': cfg['latent_space']['redundancy_factor'],
        'ecc_type': cfg['latent_space']['ecc_type'],
        'encoder': cfg.get('model', {}).get('encoder_type', 'mlp'), 'mode': mode,
        'objective': checkpoint['metadata']['objective'], 'epochs': cfg['training']['epochs'],
        'latent_source': 'optimized' if mode == 'direct' else 'encoder',
        'nmi': nmi, 'acc': acc, 'optimized_nmi': opt_nmi, 'optimized_acc': opt_acc,
        'alignment_shift_mean': float(alignment.mean()) if alignment is not None else np.nan,
        'alignment_shift_std': float(alignment.std()) if alignment is not None else np.nan,
        'robustness_shift_mean': shift,
        'robustness_relative_shift': shift / max(float(np.linalg.norm(z, axis=1).mean()), 1e-8),
        'masked_acc_mean': float(np.mean([v['acc'] for v in missing_metrics.values()])) if missing_metrics else np.nan,
        'masked_nmi_mean': float(np.mean([v['nmi'] for v in missing_metrics.values()])) if missing_metrics else np.nan,
        'avg_recon_error': float(np.mean(list(recon.values()))) if recon else np.nan,
        'z_mean_norm': float(np.linalg.norm(z, axis=1).mean()), 'z_std': float(z.std()),
        'missing_view_metrics': missing_metrics,
    }
    result.update(recon)
    return result


def write_results(results, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    details = {row['checkpoint']: {'experiment': row['experiment'], 'views': row['missing_view_metrics']}
               for row in results}
    frame = pd.DataFrame([{key: value for key, value in row.items() if key != 'missing_view_metrics'}
                          for row in results])
    groups = ['dataset', 'data_fingerprint', 'source_sha256', 'evaluation_seed',
              'latent_dim', 'mode', 'encoder', 'ecc_type', 'redundancy', 'variant_id']
    metrics = ['acc', 'nmi', 'optimized_acc', 'optimized_nmi', 'alignment_shift_mean',
               'robustness_shift_mean', 'masked_acc_mean', 'masked_nmi_mean', 'avg_recon_error']
    # A repeated run with the same training seed is a reproducibility check, not
    # another independent seed. Refuse inconsistent repeats instead of averaging them away.
    for _, repeats in frame.groupby(groups + ['seed'], dropna=False):
        if len(repeats) > 1:
            values = repeats[metrics].to_numpy(dtype=float)
            if not np.allclose(values, values[0], rtol=0, atol=0, equal_nan=True):
                raise ValueError('Repeated training seed produced inconsistent metrics')
    independent = frame.drop_duplicates(groups + ['seed'])
    frame.to_csv(output / 'comprehensive_metrics.csv', index=False)
    summary = independent.groupby(groups, dropna=False)[metrics].agg(['mean', 'std', 'count'])
    summary.columns = ['_'.join(column) for column in summary.columns]
    summary = summary.join(independent.groupby(groups)['seed'].nunique().rename('n_seeds')).reset_index()
    summary.to_csv(output / 'summary.csv', index=False)
    (output / 'metric_definitions.json').write_text(json.dumps(METRIC_DEFINITIONS, indent=2) + '\n')
    (output / 'missing_view_metrics.json').write_text(json.dumps(details, indent=2) + '\n')
    for (dataset, dim), subset in summary.groupby(['dataset', 'latent_dim']):
        labels = [f"{row['mode']}\n{row['ecc_type']} L={row['redundancy']}"
                  for _, row in subset.iterrows()]
        x = np.arange(len(subset))
        fig, axes = plt.subplots(1, 2, figsize=(max(10, 2.4 * len(subset)), 5))
        for ax, metric, title in zip(axes, ['acc', 'masked_acc_mean'],
                                     ['Full-view clustering accuracy', 'Leave-one-view-out accuracy']):
            values = subset[f'{metric}_mean'].to_numpy()
            errors = subset[f'{metric}_std'].fillna(0).to_numpy()
            valid = np.isfinite(values)
            ax.bar(x[valid], values[valid], yerr=errors[valid], capsize=4)
            ax.set_xticks(x, labels, rotation=20, ha='right', fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_title(title)
        fig.suptitle(f'{dataset}, Z={dim}; mean and sample SD across training seeds')
        fig.tight_layout()
        fig.savefig(output / f'{dataset}_Z{dim}_comparison.png', dpi=160)
        plt.close(fig)
    return frame, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', default='experiments/*')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--evaluation-seed', type=int, default=42)
    parser.add_argument('--skip-robustness', action='store_true')
    args = parser.parse_args()
    device = get_device(args.device)
    folders = []
    for path in sorted(glob.glob(args.experiments)):
        path = Path(path)
        if (path / 'config.yaml').exists():
            folders.append(path)
        elif path.is_dir():
            folders.extend(sorted(config.parent for config in path.glob('*/config.yaml')))
    if not folders:
        raise SystemExit('No experiment directories with config.yaml found')
    results, failures = [], []
    for folder in folders:
        try:
            result = analyze_experiment(folder, device, args.evaluation_seed,
                                        not args.skip_robustness)
            results.append(result)
            print(f"{result['experiment']}: ACC={result['acc']:.4f}, NMI={result['nmi']:.4f}")
        except Exception as error:
            failures.append({'experiment': str(folder), 'error': str(error)})
            print(f'FAILED {folder}: {error}')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if results:
        try:
            write_results(results, output)
        except Exception as error:
            failures.append({'stage': 'aggregation', 'error': str(error)})
            print(f'FAILED aggregation: {error}')
    (output / 'failures.json').write_text(json.dumps(failures, indent=2) + '\n')
    if failures or not results:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
