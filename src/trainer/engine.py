"""Deterministic training for all five CNG inference modes."""

import argparse
import csv
import json
import os
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils.data_utils import get_dataset
from src.utils.experiment import CHECKPOINT_VERSION, build_model, config_digest
from src.utils.helpers import dataset_fingerprint, get_device, set_seed, source_fingerprint


class Trainer:
    def __init__(self, config_path, checkpoint_dir=None, log_dir=None, experiment_dir=None):
        self.config_path = Path(config_path)
        with self.config_path.open() as handle:
            self.cfg = yaml.safe_load(handle)
        self.seed = self.cfg['experiment'].get('seed', 42)
        self.deterministic = self.cfg['experiment'].get('deterministic', True)
        set_seed(self.seed, self.deterministic)
        training = self.cfg['training']
        if training['epochs'] < 1 or training['batch_size'] < 2:
            raise ValueError('epochs must be positive and batch_size must be at least 2')
        if 'num_threads' in training:
            torch.set_num_threads(training['num_threads'])
        self.device = get_device(self.cfg['experiment'].get('device', 'auto'))
        self.exp_name = self.cfg['experiment']['name']
        dataset_name = self.cfg['experiment'].get('dataset', 'mfeat')
        timestamp = datetime.now().strftime('%Y%b%d_%H-%M-%S-%f')
        if experiment_dir:
            self.log_dir = Path(experiment_dir) / 'logs'
            self.ckpt_dir = Path(experiment_dir) / 'checkpoints'
        else:
            self.log_dir = Path(log_dir or f'logs/{dataset_name}') / f'{self.exp_name}_{timestamp}'
            self.ckpt_dir = Path(checkpoint_dir or f'checkpoints/{dataset_name}') / f'{self.exp_name}_{timestamp}'
        if (self.ckpt_dir / 'final_model.pth').exists():
            raise FileExistsError(f'Completed experiment already exists: {self.ckpt_dir}')
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        with (self.ckpt_dir / 'config.yaml').open('w') as handle:
            yaml.safe_dump(self.cfg, handle)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        print(f'[Trainer] Device: {self.device}, seed: {self.seed}, deterministic: {self.deterministic}')
        print(f'[Trainer] Checkpoints: {self.ckpt_dir}')

    def prepare_data(self):
        name = self.cfg['experiment'].get('dataset', 'mfeat')
        self.dataset = get_dataset(name, **self.cfg.get('dataset_kwargs', {}))
        if self.dataset.metadata.get('is_mock', False):
            raise ValueError('Training refuses synthetic/mock data; use real dataset files')
        self.num_data = len(self.dataset)
        if self.num_data < 2:
            raise ValueError('At least two samples are required')
        self.batch_size = min(self.cfg['training']['batch_size'], self.num_data)
        # Preserve every sample without sending a singleton batch through BatchNorm.
        while self.num_data % self.batch_size == 1:
            self.batch_size += 1
        generator = torch.Generator().manual_seed(self.seed)
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True, generator=generator,
            num_workers=self.cfg['training'].get('num_workers', 0),
            pin_memory=self.device.type == 'cuda', drop_last=False,
        )
        self.view_dims = {key: int(value[0].numel()) for key, value in self.dataset.views.items()}
        self.data_fingerprint = dataset_fingerprint(self.dataset)
        print(f'[Trainer] N={self.num_data}, effective batch size={self.batch_size}, views={self.view_dims}')

    def build_model(self):
        self.model = build_model(self.cfg, self.dataset, self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg['training']['lr'])
        self.amp_dtype = torch.bfloat16 if self.device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
        requested = self.cfg['training'].get('use_amp', False) and self.device.type == 'cuda'
        # FP16's range is insufficient for GP gradients; BF16 or FP32 preserves it.
        gp_loss = self.cfg['training'].get('use_gp_loss', True)
        self.use_amp = requested and (not gp_loss or self.amp_dtype == torch.bfloat16)
        self.scaler = GradScaler('cuda', enabled=self.use_amp and self.amp_dtype == torch.float16)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = total_data = total_kl = total_align = 0.0
        training = self.cfg['training']
        progress = tqdm(self.dataloader, desc=f'Epoch {epoch}', leave=False,
                        disable=not training.get('progress', True))
        for views, _, indices in progress:
            views = {key: value.to(self.device).float() for key, value in views.items()}
            indices = indices.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
                loss, details = self.model.compute_loss(
                    views, indices, beta=training.get('beta', 1.0),
                    use_gp_loss=training.get('use_gp_loss', True),
                    alignment_beta=training.get('alignment_beta', 0.1),
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at epoch {epoch}')
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                          training.get('max_grad_norm', 1.0),
                                          error_if_nonfinite=True)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
            total_data += details['data_loss']
            total_kl += details['kl_loss']
            total_align += details.get('align_loss', 0.0)
            progress.set_postfix(loss=loss.item() / len(indices))
        self.last_alignment_loss = total_align / self.num_data
        return total_loss / self.num_data, total_data / self.num_data, total_kl / self.num_data

    def fit(self):
        history = []
        try:
            self.prepare_data()
            self.build_model()
            training = self.cfg['training']
            for epoch in range(1, training['epochs'] + 1):
                loss, data_loss, kl = self.train_epoch(epoch)
                record = dict(epoch=epoch, loss=loss, data_loss=data_loss, kl=kl,
                              alignment_loss=self.last_alignment_loss)
                history.append(record)
                for key, value in record.items():
                    if key != 'epoch':
                        self.writer.add_scalar(f'Loss/{key}', value, epoch)
                if epoch % training.get('log_interval', 10) == 0 or epoch == training['epochs']:
                    print(f'Epoch {epoch}: loss={loss:.6f}, data={data_loss:.6f}, KL={kl:.6f}')
            metadata = {
                'format_version': CHECKPOINT_VERSION, 'seed': self.seed,
                'deterministic': self.deterministic, 'device': str(self.device),
                'amp_enabled': self.use_amp, 'amp_dtype': str(self.amp_dtype) if self.use_amp else None,
                'config_sha256': config_digest(self.cfg), 'data_fingerprint': self.data_fingerprint,
                'source_sha256': source_fingerprint(),
                'data_metadata': self.dataset.metadata, 'num_samples': self.num_data,
                'effective_batch_size': self.batch_size,
                'objective': ('full_gp_marginal' if self.batch_size == self.num_data else 'batch_gp_marginal')
                             if training.get('use_gp_loss', True) else 'gaussian_reconstruction',
                'versions': {name: version(name) for name in
                             ('torch', 'numpy', 'scipy', 'scikit-learn', 'PyYAML')},
            }
            save_path = self.ckpt_dir / 'final_model.pth'
            torch.save({'format_version': CHECKPOINT_VERSION, 'config': self.cfg,
                        'metadata': metadata, 'model_state_dict': self.model.state_dict()}, save_path)
            (self.ckpt_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
            with (self.ckpt_dir / 'history.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
            print(f'[Trainer] Saved {save_path}')
            return save_path
        finally:
            self.writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint_dir')
    parser.add_argument('--log_dir')
    parser.add_argument('--experiment_dir')
    args = parser.parse_args()
    Trainer(args.config, checkpoint_dir=args.checkpoint_dir, log_dir=args.log_dir,
            experiment_dir=args.experiment_dir).fit()
