# src/trainer/engine.py

import os
# Fix for Mac OpenMP duplicate library error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from datetime import datetime
try:
    from torch.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    try:
        from torch.cuda.amp import autocast, GradScaler
        AMP_AVAILABLE = True
    except ImportError:
        AMP_AVAILABLE = False

# Import Modules
from src.utils.data_utils import get_dataset, load_mfeat_data
from src.models.cng_model import CNG_MV_GPLVM

class Trainer:
    def __init__(self, config_path, checkpoint_dir=None, log_dir=None, experiment_dir=None):
        self.config_path = config_path
        self.experiment_dir = experiment_dir
        # 1. Load Configuration
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        # Setup Device
        if self.cfg['experiment']['device'] == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"[Trainer] Using CUDA acceleration.")
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
                print(f"[Trainer] Using MPS (Apple Silicon) acceleration.")
            else:
                self.device = torch.device('cpu')
                print(f"[Trainer] Using CPU.")
        else:
            self.device = torch.device(self.cfg['experiment']['device'])
            
        # Optimization: CUDNN Benchmark
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            print("[Trainer] Enabled CUDNN benchmark.")
            
        # Logging - Organized by Dataset or Override
        self.exp_name = self.cfg['experiment']['name']
        self.dataset_name = self.cfg['experiment'].get('dataset', 'unknown')
        current_time = datetime.now().strftime('%Y%b%d_%H-%M-%S')
        
        # Determine paths
        if self.experiment_dir:
            # New Workflow: Clean paths inside experiment_dir
            self.log_dir = os.path.join(self.experiment_dir, 'logs')
            self.ckpt_dir = os.path.join(self.experiment_dir, 'checkpoints')
        else:
            # Legacy Workflow: Append timestamps
            if log_dir:
                 self.log_dir = os.path.join(log_dir, f"{self.exp_name}_{current_time}")
            else:
                 self.log_dir = os.path.join('logs', self.dataset_name, f"{self.exp_name}_{current_time}")

            if checkpoint_dir:
                self.ckpt_dir = os.path.join(checkpoint_dir, f"{self.exp_name}_{current_time}")
            else:
                self.ckpt_dir = os.path.join('checkpoints', self.dataset_name, f"{self.exp_name}_{current_time}")

        self.writer = SummaryWriter(log_dir=self.log_dir)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        # Save config for reproducibility and automated analysis
        with open(os.path.join(self.ckpt_dir, 'config.yaml'), 'w') as f:
            yaml.dump(self.cfg, f)
            
        print(f"[Trainer] Log Dir: {self.log_dir}")
        print(f"[Trainer] Ckpt Dir: {self.ckpt_dir}")
        
    def prepare_data(self):
        # 默认模式为 real
        print("[Trainer] Loading Dataset...")
        dataset_name = self.cfg['experiment'].get('dataset', 'mfeat')
        print(f"[Trainer] Loading {dataset_name}...")
        
        # Extract dataset-specific kwargs if present
        dataset_kwargs = self.cfg.get('dataset_kwargs', {})
        
        try:
            self.dataset = get_dataset(dataset_name, **dataset_kwargs)
        except Exception as e:
            print(f"[Trainer] Error loading dataset {dataset_name}: {e}")
            if dataset_name == 'mfeat': # Specific fallback for mfeat download
                 print("[Trainer] Attempting to download mfeat...")
                 from src.utils.data_download import download_mfeat
                 download_mfeat()
                 self.dataset = get_dataset('mfeat')
            else:
                 raise e
        
        # DataLoader Optimization
        self.batch_size = self.cfg['training']['batch_size']
        num_workers = self.cfg['training'].get('num_workers', 4 if self.device.type == 'cuda' else 0)
        pin_memory = True if self.device.type == 'cuda' else False
        
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        
        # Extract view dims
        self.view_dims = {k: v.shape[1] for k, v in self.dataset.views.items()}
        self.num_data = len(self.dataset)
        print(f"[Trainer] Data Loaded. N={self.num_data}, Views={self.view_dims}")

    def build_model(self):
        print("[Trainer] Building Model...")
        feature_cfg = self.cfg['latent_space']
        kernel_cfg = self.cfg['kernels']
        
        # 显式控制 use_ecc
        use_ecc = True
        if 'ecc_type' in feature_cfg and feature_cfg['ecc_type'] == 'none':
            use_ecc = False
        
        # Read Inference Mode (New param)
        inference_mode = 'direct'
        if 'model' in self.cfg and 'inference_mode' in self.cfg['model']:
            inference_mode = self.cfg['model']['inference_mode']
            
        # Read Encoder Type
        encoder_type = 'mlp'
        if 'model' in self.cfg and 'encoder_type' in self.cfg['model']:
            encoder_type = self.cfg['model']['encoder_type']
            
        # Get ECC Mode
        ecc_mode = feature_cfg.get('ecc_type', 'repetition')
        
        # Extract view_shapes for image-based datasets (optional)
        # Check both singular and plural to prevent configuration mismatch
        view_shapes = self.cfg.get('view_shapes', self.cfg.get('views_shapes', None))
        
        # Decide whether to use heavy readout layers (save memory for high-dim inputs)
        use_readouts = True
        if view_shapes is not None or any(v.get('input_dim', 0) > 10000 for v in self.view_dims.values()):
             # Image inputs (usually >10k dims) are too large for Linear Readouts (8k -> 82k params)
             # Force disable readouts to save ~10GB VRAM per process
             use_readouts = False
             print(f"[Trainer] High-dimensional view detected. Disabling Readout layers (VRAM Optimized).")
            
        self.model = CNG_MV_GPLVM(
            num_data=self.num_data,
            input_dim=feature_cfg['info_dim'],
            view_dims=self.view_dims,
            redundancy_factor=feature_cfg['redundancy_factor'],
            use_ecc=use_ecc,
            num_mixtures=kernel_cfg['num_mixtures'],
            rff_samples=kernel_cfg['rff_samples'],
            ecc_matrix_path=None,
            inference_mode=inference_mode,
            ecc_mode=ecc_mode,
            encoder_type=encoder_type,
            view_shapes=view_shapes,
            use_readouts=use_readouts
        ).to(self.device)
        
        # Optimizer
        self.lr = self.cfg['training']['lr']
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # AMP Scaler
        self.use_amp = self.cfg['training'].get('use_amp', True) and self.device.type == 'cuda' and AMP_AVAILABLE
        self.scaler = GradScaler() if self.use_amp else None
        
        print(f"[Trainer] Model built on {self.device}. (AMP: {self.use_amp})")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch}", leave=False)
        
        for batch_idx, (views_batch, labels, indices) in enumerate(progress_bar):
            # Move data to device
            views_batch = {k: v.to(self.device).float() for k, v in views_batch.items()}
            indices = indices.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward + Loss with AMP
            use_gp_loss = self.cfg.get('training', {}).get('use_gp_loss', True)
            
            if self.use_amp:
                with autocast(device_type='cuda'):
                    loss, details = self.model.compute_loss(views_batch, indices, beta=1.0, use_gp_loss=use_gp_loss)
                    
                    # --- Alignment Loss (Semi-Amortized) ---
                    if self.model.inference_mode == 'semi_amortized':
                        outputs = self.model(indices, views_batch)
                        mu_opt, mu_enc = outputs[1], outputs[3]
                        align_loss = torch.nn.functional.mse_loss(mu_enc, mu_opt.detach(), reduction='sum')
                        align_beta = self.cfg.get('training', {}).get('alignment_beta', 0.1)
                        loss = loss + align_beta * align_loss
                        details['align_loss'] = align_loss.item()
                
                self.scaler.scale(loss).backward()
                
                # Unscale for Clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, details = self.model.compute_loss(views_batch, indices, beta=1.0, use_gp_loss=use_gp_loss)
                
                if self.model.inference_mode == 'semi_amortized':
                    outputs = self.model(indices, views_batch)
                    mu_opt, mu_enc = outputs[1], outputs[3]
                    align_loss = torch.nn.functional.mse_loss(mu_enc, mu_opt.detach(), reduction='sum')
                    align_beta = self.cfg.get('training', {}).get('alignment_beta', 0.1)
                    loss = loss + align_beta * align_loss
                    details['align_loss'] = align_loss.item()
                
                loss.backward()
                
                # NaN Guard: Check for parameter corruption
                has_nan = False
                for name, param in self.model.named_parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        print(f"[CRITICAL] NaN/Inf gradient detected in {name}")
                        has_nan = True
                        break
                
                if has_nan:
                    # Skip this batch to prevent corruption
                    self.optimizer.zero_grad()
                    continue
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Stats
            batch_size = len(indices)
            total_loss += loss.item()
            total_recon += details.get('data_loss', details.get('recon_loss', 0))
            total_kl += details.get('kl_loss', 0)
            
            postfix_dict = {'loss': loss.item() / batch_size}
            if 'align_loss' in details:
                postfix_dict['align'] = details['align_loss'] / batch_size
                
            progress_bar.set_postfix(postfix_dict)
            
        # Epoch Summary
        avg_loss = total_loss / self.num_data
        avg_recon = total_recon / self.num_data
        avg_kl = total_kl / self.num_data
        
        return avg_loss, avg_recon, avg_kl

    def fit(self):
        self.prepare_data()
        self.build_model()
        
        epochs = self.cfg['training']['epochs']
        log_interval = self.cfg['training']['log_interval']
        
        print("\n[Trainer] Start Training...")
        for epoch in range(1, epochs + 1):
            avg_loss, avg_recon, avg_kl = self.train_epoch(epoch)
            
            # TensorBoard
            self.writer.add_scalar('Loss/Total', avg_loss, epoch)
            self.writer.add_scalar('Loss/Recon', avg_recon, epoch)
            self.writer.add_scalar('Loss/KL', avg_kl, epoch)
            
            if epoch % log_interval == 0:
                print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, KL: {avg_kl:.4f})")
                
                # Check metrics (simple noise sigma check)
                noise_vals = {k: torch.exp(v).item() for k, v in self.model.log_noise_sigmas.items()}
                # print(f"  -> Noise Sigmas: {noise_vals}")
                
        # Save Final Model
        save_path = os.path.join(self.ckpt_dir, "final_model.pth")
        torch.save(self.model.state_dict(), save_path)
        print(f"[Trainer] Saved final model to {save_path}")
        self.writer.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/mfeat_default.yaml', help='Path to config file')
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Override checkpoint directory")
    parser.add_argument("--log_dir", type=str, default=None, help="Override log directory")
    parser.add_argument("--experiment_dir", type=str, default=None, help="Root directory for the experiment (contains config, logs, checkpoints)")
    args = parser.parse_args()
    
    trainer = Trainer(args.config, checkpoint_dir=args.checkpoint_dir, log_dir=args.log_dir, experiment_dir=args.experiment_dir)
    trainer.fit()