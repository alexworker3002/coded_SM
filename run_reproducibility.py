#!/usr/bin/env python
"""Small real-data controls; validation of execution, not a performance benchmark."""

import argparse
import copy
import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from auto_analyze import analyze_experiment, write_results
from src.trainer.engine import Trainer
from src.utils.data_utils import get_dataset
from src.utils.helpers import dataset_fingerprint, set_seed, source_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['mfeat', '100leaves', 'caltech101-7'], default='mfeat')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--run-dir', default='experiments/reproducibility')
    parser.add_argument('--output', default='reports/reproducibility')
    args = parser.parse_args()
    initial_source = source_fingerprint()
    if len(set(args.seeds)) < 2 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Supply at least two distinct training seeds')
    run_dir, output = Path(args.run_dir), Path(args.output)
    if run_dir.exists() or output.exists():
        raise FileExistsError('Choose fresh --run-dir and --output folders for a new comparison')
    set_seed(42)
    torch.set_num_threads(1)
    data = get_dataset(args.dataset)
    if data.metadata.get('is_mock', False):
        raise ValueError('Real data is required')
    variants = [
        ('semi_none', 'semi_amortized', 'none', 1),
        ('semi_rep_L2', 'semi_amortized', 'repetition', 2),
        ('semi_random_L2', 'semi_amortized', 'random_gaussian', 2),
        ('coded_semi_rep_L2', 'coded_semi_amortized', 'repetition', 2),
        ('coded_semi_random_L2', 'coded_semi_amortized', 'random_gaussian', 2),
    ]
    results, reference = [], None
    for label, mode, ecc, redundancy in variants:
        for seed in args.seeds:
            folder = run_dir / f'{label}_seed{seed}'
            folder.mkdir(parents=True)
            cfg = {
                'experiment': {'name': folder.name, 'dataset': args.dataset, 'device': args.device,
                               'seed': seed, 'deterministic': True},
                'latent_space': {'info_dim': 10, 'redundancy_factor': redundancy, 'ecc_type': ecc},
                'model': {'inference_mode': mode, 'encoder_type': 'mlp'},
                'kernels': {'type': 'ng_sm', 'num_mixtures': 2, 'rff_samples': 32},
                'training': {'batch_size': len(data), 'epochs': args.epochs, 'lr': 0.001,
                             'use_gp_loss': True, 'alignment_beta': 0.1, 'use_amp': False,
                             'num_workers': 0, 'num_threads': 1, 'progress': False,
                             'log_interval': args.epochs},
            }
            path = folder / 'config.yaml'
            path.write_text(yaml.safe_dump(cfg))
            trainer = Trainer(path, experiment_dir=folder)
            trainer.fit()
            metrics = analyze_experiment(trainer.ckpt_dir, device=args.device, dataset=data)
            results.append(metrics)
            print(f"[Control] {folder.name}: ACC={metrics['acc']:.4f}, NMI={metrics['nmi']:.4f}", flush=True)
            if reference is None:
                reference = (copy.deepcopy(cfg), {key: value.detach().cpu().clone()
                             for key, value in trainer.model.state_dict().items()}, metrics.copy(),
                             (trainer.ckpt_dir / 'history.csv').read_bytes())
            del trainer
    cfg, expected_state, expected_metrics, expected_history = reference
    repeat = run_dir / 'same_seed_repeat'
    repeat.mkdir()
    repeat_path = repeat / 'config.yaml'
    repeat_path.write_text(yaml.safe_dump(cfg))
    trainer = Trainer(repeat_path, experiment_dir=repeat)
    trainer.fit()
    actual_state = trainer.model.state_dict()
    state_equal = all(torch.equal(value, actual_state[key].detach().cpu())
                      for key, value in expected_state.items())
    history_equal = expected_history == (trainer.ckpt_dir / 'history.csv').read_bytes()
    repeat_metrics = analyze_experiment(trainer.ckpt_dir, device=args.device, dataset=data)
    measured_keys = ['acc', 'nmi', 'alignment_shift_mean', 'robustness_shift_mean', 'masked_acc_mean']
    metrics_equal = all(expected_metrics[key] == repeat_metrics[key] for key in measured_keys)
    if source_fingerprint() != initial_source:
        raise RuntimeError('Source files changed during validation; run again with stable source files')
    frame, summary = write_results(results, output)
    packages = ('torch', 'numpy', 'scipy', 'scikit-learn', 'PyYAML', 'pandas',
                'matplotlib', 'tensorboard', 'tqdm', 'Pillow', 'threadpoolctl')
    manifest = {
        'purpose': 'Short transductive real-data validation; insufficient to establish ECC superiority.',
        'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': source_fingerprint(), 'python': platform.python_version(),
        'torch_build': str(torch.__version__), 'device': args.device,
        'versions': {name: version(name) for name in packages},
        'data_fingerprint': dataset_fingerprint(data), 'data_metadata': data.metadata,
        'dataset': args.dataset, 'num_samples': len(data),
        'view_dims': {key: int(value[0].numel()) for key, value in data.views.items()},
        'epochs': args.epochs, 'training_seeds': args.seeds, 'evaluation_seed': 42,
        'objective': 'Full-data GP marginal likelihood plus KL/(N*50) and mean alignment.',
        'same_seed_repeat': {'all_model_tensors_identical': state_equal,
                             'history_bytes_identical': history_equal,
                             'metrics_identical': metrics_equal,
                             'excluded_from_seed_summary': True},
        'num_control_runs': len(results), 'num_total_trainings': len(results) + 1,
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'requirements-validated.txt').write_text(
        '\n'.join(f'{name}=={version(name)}' for name in packages) + '\n')
    if not state_equal or not history_equal or not metrics_equal:
        raise AssertionError(f'Same-seed reproducibility failed: {manifest["same_seed_repeat"]}')
    lines = [
        '# Real-data reproducibility check', '',
        f'- Dataset: {args.dataset}, N={len(data)}, all available views.',
        f'- Controls: five variants, seeds {args.seeds}, {args.epochs} epochs each; one extra same-seed repeat.',
        '- Identical training budget: Z=10, Q=2, 32 RFF samples per mixture, full-data batches, LR=0.001.',
        '- Evaluation: transductive KMeans (20 starts, seed 42), encoder means, leave-one-view-out robustness.',
        '- All repeated model tensors, loss histories and evaluated metrics match exactly.',
        '- GP readouts are disabled; reconstruction MSE is unavailable, not zero.',
        '- This short run validates the repaired pipeline and does not establish a performance advantage.', '',
        '| Mode | ECC | Seeds | ACC mean ± SD | NMI mean ± SD | Masked ACC mean ± SD |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for _, row in summary.iterrows():
        lines.append(f"| {row['mode']} | {row['ecc_type']} L={row['redundancy']} | {row['n_seeds']} | "
                     f"{row['acc_mean']:.4f} ± {row['acc_std']:.4f} | "
                     f"{row['nmi_mean']:.4f} ± {row['nmi_std']:.4f} | "
                     f"{row['masked_acc_mean_mean']:.4f} ± {row['masked_acc_mean_std']:.4f} |")
    lines += ['', 'Per-run metrics, missing-view details, metric definitions and the source/data/version manifest',
              'are stored alongside this report. Checkpoints and training configs were written locally to '
              + str(run_dir) + ' (ignored by Git); use the command below to recreate them.',
              '', 'Reproduce in fresh directories:', '', '```bash',
              f'python run_reproducibility.py --dataset {args.dataset} --epochs {args.epochs} '
              f'--seeds {" ".join(map(str, args.seeds))} --device {args.device} '
              '--run-dir experiments/repro-repeat --output reports/repro-repeat', '```', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))
    print(summary[['mode', 'ecc_type', 'acc_mean', 'acc_std', 'nmi_mean', 'nmi_std']].to_string(index=False))
    print(f'[Verified] Exact same-seed repeat. Report: {output / "REPORT.md"}')


if __name__ == '__main__':
    main()
