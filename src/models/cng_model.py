# src/models/cng_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 导入模块 (根据实际路径调整)
from src.modules.ecc import LinearECCProjection
from src.kernels.ng_sm_kernel import NextGenSpectralMixtureKernel

class CNG_MV_GPLVM(nn.Module):
    """
    Coded Next-Gen Multi-View GPLVM (CNG-MV-GPLVM)
    
    该模型整合了:
    1. Variational Inference for Latents Z
    2. Linear Error Correcting Codes (ECC) for structural regularization
    3. Next-Gen Spectral Mixture Kernel (NG-SM) with RFF approximation
    
    设计支持 Ablation Study:
    - 可以通过 toggle `use_ecc` 关闭编码模块，退化为通过 NG-SM 核的普通 GPLVM。
    """
    def __init__(self, 
                 num_data,        # N: 样本总数
                 input_dim,       # d_z: 潜变量 Z 的维度
                 view_dims,       # Dict[str, int]: 各视图的名称和维度
                 redundancy_factor=2, # L: 编码冗余度
                 use_ecc=True,    # Switch: 是否启用 ECC
                 num_mixtures=4,  # Q: 核函数混合成分数
                 rff_samples=100, # S: RFF 采样数
                 z_init_std=0.01, # q(z) 初始标准差
                 ecc_matrix_path=None # 预定义的 ECC 矩阵路径
                 ):
        super().__init__()
        
        self.num_data = num_data
        self.input_dim = input_dim
        self.view_dims = view_dims
        self.use_ecc = use_ecc
        self.redundancy_factor = redundancy_factor
        
        # =========================================================
        # 1. Variational Parameters for Latent Z: q(Z)
        # =========================================================
        self.q_mu = nn.Parameter(torch.randn(num_data, input_dim) * z_init_std)
        self.q_log_sigma = nn.Parameter(torch.ones(num_data, input_dim) * np.log(z_init_std))
        
        # =========================================================
        # 2. ECC Module (Optional)
        # =========================================================
        if self.use_ecc:
            self.x_dim = input_dim * redundancy_factor
            self.ecc_module = LinearECCProjection(input_dim, redundancy_factor, matrix_path=ecc_matrix_path)
            print(f"[Model] Initialized with ECC. Z({input_dim}) -> X({self.x_dim})")
        else:
            self.x_dim = input_dim
            self.ecc_module = nn.Identity()
            print(f"[Model] Initialized WITHOUT ECC (Baseline Mode). Z({input_dim}) -> X({self.x_dim})")

        # =========================================================
        # 3. Multi-View Components (Kernels + Readouts)
        # =========================================================
        # 为每个视图建立独立的 NG-SM 核、Readout 层和噪声参数
        # 共享: Z, ECC投影
        # 独立: Kernel Spectrum, Readout Weights, Noise Variance
        
        self.kernels = nn.ModuleDict()
        self.readouts = nn.ModuleDict()
        self.log_noise_sigmas = nn.ParameterDict()
        
        # RFF 输出维度 (Bias-free Concatenated Cos/Sin): 2 * Q * S
        self.feature_dim = 2 * num_mixtures * rff_samples
        
        for name, v_dim in view_dims.items():
            # Kernel
            self.kernels[name] = NextGenSpectralMixtureKernel(
                num_dims=self.x_dim,
                num_mixtures=num_mixtures, 
                rff_samples=rff_samples
            )
            # Readout
            self.readouts[name] = nn.Linear(self.feature_dim, v_dim, bias=True)
            # Noise (init log(-2) ~ 0.135)
            self.log_noise_sigmas[name] = nn.Parameter(torch.tensor(-2.0))

    def reparameterize(self, mu, log_sigma):
        """
        z = mu + sigma * epsilon
        """
        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def get_latents(self, batch_indices=None):
        if batch_indices is None:
            return self.q_mu, self.q_log_sigma
        else:
            return self.q_mu[batch_indices], self.q_log_sigma[batch_indices]

    def forward(self, batch_indices):
        """
        前向传播
        Returns: 
            y_recons (dict): {view_name: recon_batch}
            batch_mu
            batch_log_sigma
        """
        # 1. 采样 Z
        batch_mu, batch_log_sigma = self.get_latents(batch_indices)
        z_sample = self.reparameterize(batch_mu, batch_log_sigma)
        
        # 2. ECC 编码: Z -> X
        x_sample = self.ecc_module(z_sample)
        
        # 3. Multi-View Mapping
        y_recons = {}
        
        for name in self.view_dims.keys():
            # Kernel Mapping: X -> Phi_v(X)
            features = self.kernels[name].get_rff_feature(x_sample)
            # Readout: Phi_v -> Y_hat_v
            y_recons[name] = self.readouts[name](features)
        
        return y_recons, batch_mu, batch_log_sigma

    def compute_loss(self, views_batch, batch_indices, beta=1.0):
        """
        views_batch: dict {view_name: tensor}
        """
        y_recons, mu, log_sigma = self.forward(batch_indices)
        
        total_recon_loss = 0.0
        details = {}
        
        # --- 1. Reconstruction Loss (Sum over views) ---
        for name, y_true in views_batch.items():
            y_pred = y_recons[name]
            noise_sigma = torch.exp(self.log_noise_sigmas[name])
            
            # NLL
            mse = (y_true - y_pred).pow(2)
            nll = torch.log(noise_sigma) + 0.5 * mse / (noise_sigma ** 2)
            view_loss = nll.sum()
            
            total_recon_loss += view_loss
            details[f"recon_{name}"] = view_loss.item()
            details[f"sigma_{name}"] = noise_sigma.item()
        
        # --- 2. KL Divergence for Z (Shared) ---
        var = torch.exp(2 * log_sigma)
        kl_div = -0.5 * torch.sum(1 + 2 * log_sigma - mu.pow(2) - var)
        
        details["kl_loss"] = kl_div.item()
        details["recon_loss"] = total_recon_loss.item() # Fix for KeyError
        
        return total_recon_loss + beta * kl_div, details
