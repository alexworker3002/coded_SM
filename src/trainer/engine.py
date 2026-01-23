# src/trainer/engine.py

import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Import Modules
from src.utils.data_utils import load_mfeat_data
from src.models.cng_model import CNG_MV_GPLVM

class Trainer:
    def __init__(self, config_path):
        # 1. Load Configuration
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        # Setup Device
        if self.cfg['experiment']['device'] == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
                print(f"[Trainer] Using MPS (Apple Silicon) acceleration.")
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"[Trainer] Using CUDA acceleration.")
            else:
                self.device = torch.device('cpu')
                print(f"[Trainer] Using CPU.")
        else:
            self.device = torch.device(self.cfg['experiment']['device'])
            
        # Logging
        self.exp_name = self.cfg['experiment']['name']
        current_time = datetime.now().strftime('%Y%b%d_%H-%M-%S')
        log_dir = os.path.join('logs', f"{self.exp_name}_{current_time}")
        self.writer = SummaryWriter(log_dir=log_dir)
        self.ckpt_dir = os.path.join('checkpoints', f"{self.exp_name}_{current_time}")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        print(f"[Trainer] Log Dir: {log_dir}")
        print(f"[Trainer] Ckpt Dir: {self.ckpt_dir}")
        
    def prepare_data(self):
        # Load Dataset
        # 默认模式为 real
        print("[Trainer] Loading Dataset...")
        # 注意: load_mfeat_data 内部可能会 check 文件是否存在，如果不在会报错
        # 我们假设已下载
        try:
            self.dataset = load_mfeat_data(mode="real")
        except FileNotFoundError:
            print("[Trainer] Real data not found, falling back to MOCK mode.")
            self.dataset = load_mfeat_data(mode="mock")
        
        # DataLoader
        self.batch_size = self.cfg['training']['batch_size']
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=0, # MPS 兼容性
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
            
        self.model = CNG_MV_GPLVM(
            num_data=self.num_data,
            input_dim=feature_cfg['info_dim'],
            view_dims=self.view_dims,
            redundancy_factor=feature_cfg['redundancy_factor'],
            use_ecc=use_ecc,
            num_mixtures=kernel_cfg['num_mixtures'],
            rff_samples=kernel_cfg['rff_samples'],
        ).to(self.device)
        
        # Optimizer
        self.lr = self.cfg['training']['lr']
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        print(f"[Trainer] Model built on {self.device}.")

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
            
            # Forward + Loss
            # beta = 1.0 (Standard VAE), can be annealed
            loss, details = self.model.compute_loss(views_batch, indices, beta=1.0)
            
            loss.backward()
            self.optimizer.step()
            
            # Stats (scaled by batch size already in compute_loss? No, compute_loss returns SUM)
            # Typically valid batch loss to display is average
            batch_size = len(indices)
            total_loss += loss.item()
            total_recon += details['recon_loss']
            total_kl += details['kl_loss']
            
            progress_bar.set_postfix({'loss': loss.item() / batch_size})
            
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
    args = parser.parse_args()
    
    trainer = Trainer(args.config)
    trainer.fit()