# src/kernels/ng_sm_kernel.py
"""
Next-Gen Spectral Mixture Kernel (NG-SM) - Yang et al. (2025) Aligned Implementation

Key Features:
- Bivariate Gaussian spectral density with correlation parameter rho
- Two-step reparameterization trick for dynamic RFF sampling
- Per-mixture component parameters: mu1, mu2, std1, std2, rho, weight
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NextGenSpectralMixtureKernel(nn.Module):
    """
    NG-SM Kernel with Bivariate Gaussian Spectral Density.
    
    Implements the exact formulation from Yang (2025) NG-MVLVM:
    - Each mixture component q has parameters: (w_q, mu1_q, mu2_q, std1_q, std2_q, rho_q)
    - RFF features use two-step reparameterization for bivariate Gaussian sampling
    
    Parameters:
        num_dims (int): Latent dimension (Q in paper notation)
        num_mixtures (int): Number of spectral mixture components (M in paper notation)
        rff_samples (int): Number of RFF samples per component (L/2 in paper notation)
    """
    def __init__(self, num_dims, num_mixtures=4, rff_samples=50):
        super().__init__()
        self.num_dims = num_dims      # Q: latent dimension
        self.num_mixtures = num_mixtures  # M: mixture components
        self.rff_samples = rff_samples    # L/2: spectral points per component
        
        # ========== Bivariate Gaussian Parameters (per mixture) ==========
        # log_weight: M x 1 -> weight after softplus
        self.log_weight = nn.Parameter(torch.randn(num_mixtures, 1))
        
        # Spectral means: mu1, mu2 (M x Q)
        # For M=1 (SE kernel), means are fixed at 0
        if num_mixtures == 1:
            self.mu1 = nn.Parameter(torch.zeros(num_mixtures, num_dims), requires_grad=False)
            self.mu2 = nn.Parameter(torch.zeros(num_mixtures, num_dims), requires_grad=False)
        else:
            self.mu1 = nn.Parameter(torch.zeros(num_mixtures, num_dims))
            self.mu2 = nn.Parameter(torch.zeros(num_mixtures, num_dims))
        
        # Spectral stds: log_std1, log_std2 (M x Q) -> std after softplus
        self.log_std1 = nn.Parameter(torch.ones(num_mixtures, num_dims))
        self.log_std2 = nn.Parameter(torch.ones(num_mixtures, num_dims))
        
        # Correlation coefficient: rho (M,) - unbounded, will use tanh to constrain to (-1, 1)
        self.raw_rho = nn.Parameter(torch.zeros(num_mixtures))
    
    @property
    def weight(self):
        """Mixture weights (positive via softplus)."""
        return F.softplus(self.log_weight)  # M x 1
    
    @property
    def std1(self):
        """First marginal std (positive via softplus)."""
        return F.softplus(self.log_std1)  # M x Q
    
    @property
    def std2(self):
        """Second marginal std (positive via softplus)."""
        return F.softplus(self.log_std2)  # M x Q
    
    @property
    def rho(self):
        """Correlation coefficient (constrained to (-1, 1) via tanh)."""
        return torch.tanh(self.raw_rho)  # M
    
    def get_rff_feature(self, x):
        """
        Compute Random Fourier Features using Two-Step Reparameterization.
        
        Math (Yang 2025, Eq. in _compute_sm_basis):
            For each mixture component q:
                eps1 ~ N(0, I), eps2 ~ N(0, I)
                omega1 = mu1_q + std1_q * eps1
                omega2 = mu2_q + rho_q * (std2_q / std1_q) * (omega1 - mu1_q) 
                         + sqrt(1 - rho_q^2) * std1_q * eps2
                
                phi_q = sqrt(w_q / (4*S)) * [cos(2π x @ omega1.T) + cos(2π x @ omega2.T),
                                              sin(2π x @ omega1.T) + sin(2π x @ omega1.T)]
                                              
        Args:
            x: [N, Q] - Latent coordinates
            
        Returns:
            Phi: [N, M * 2 * S] - RFF features (concatenated across all mixtures)
        """
        N = x.size(0)
        device = x.device
        
        # Get transformed parameters
        w = self.weight        # M x 1
        s1 = self.std1         # M x Q
        s2 = self.std2         # M x Q
        rho = self.rho         # M
        
        all_phi = []
        
        for q in range(self.num_mixtures):
            # Sample random noise for this forward pass (dynamic sampling)
            eps1 = torch.randn(self.rff_samples, self.num_dims, device=device)  # S x Q
            eps2 = torch.randn(self.rff_samples, self.num_dims, device=device)  # S x Q
            
            # Two-step reparameterization trick for bivariate Gaussian
            # Step 1: Sample omega1 ~ N(mu1, std1^2)
            omega1 = self.mu1[q] + s1[q] * eps1  # S x Q
            
            # Step 2: Sample omega2 | omega1 (conditional Gaussian)
            # omega2 = mu2 + rho * (std2/std1) * (omega1 - mu1) + sqrt(1-rho^2) * std1 * eps2
            rho_q = rho[q]
            omega2 = (self.mu2[q] + 
                      rho_q * (s2[q] / s1[q]) * (omega1 - self.mu1[q]) + 
                      torch.sqrt(1 - rho_q ** 2) * s1[q] * eps2)  # S x Q
            
            # Compute spectral projections: x @ omega.T
            # x: [N, Q], omega: [S, Q] -> projection: [N, S]
            proj1 = 2 * np.pi * x.matmul(omega1.t())  # N x S
            proj2 = 2 * np.pi * x.matmul(omega2.t())  # N x S
            
            # Compute RFF features (Yang 2025 Eq. 149-150)
            # Phi_q = sqrt(w_q / (4*S)) * [cos(proj1) + cos(proj2), sin(proj1) + sin(proj1)]
            # Note: Reference code has sin(proj1) + sin(proj1), which seems like a bug but we follow it exactly
            amplitude = torch.sqrt(w[q] / (4 * self.rff_samples))  # scalar
            
            cos_feat = amplitude * (torch.cos(proj1) + torch.cos(proj2))  # N x S
            sin_feat = amplitude * (torch.sin(proj1) + torch.sin(proj1))  # N x S (follows reference)
            
            # Concatenate cos and sin: [N, 2*S]
            phi_q = torch.cat([cos_feat, sin_feat], dim=1)
            all_phi.append(phi_q)
        
        # Concatenate across all mixtures: [N, M * 2 * S]
        Phi = torch.cat(all_phi, dim=1)
        return Phi
    
    @property
    def feature_dim(self):
        """Output feature dimension."""
        return self.num_mixtures * 2 * self.rff_samples