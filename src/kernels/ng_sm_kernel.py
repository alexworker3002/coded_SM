
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NextGenSpectralMixtureKernel(nn.Module):
    """
    Next-Gen Spectral Mixture (NG-SM) Kernel.
    Aligned with Yang (2025) NG-MVLVM implementation.
    
    Key Features:
    1. Bivariate Gaussian Spectral Density per mixture component (mu1, mu2, std1, std2, rho).
    2. Dynamic RFF Sampling (resampled every forward pass).
    3. Specific Feature Construction (Sum of Cosines/Sines).
    """
    def __init__(self, num_dims, num_mixtures=4, rff_samples=100):
        super().__init__()
        self.num_dims = num_dims # D (Input dim, e.g. x_dim)
        self.num_mixtures = num_mixtures # M
        self.rff_samples = rff_samples # S
        
        # In reference, feature dim is M * (2 * S) 
        self._feature_dim = self.num_mixtures * 2 * self.rff_samples

        # Parameters
        # Weights (alpha): M x 1
        self.raw_weights = nn.Parameter(torch.randn(num_mixtures, 1))
        
        # Means (mu1, mu2): M x D
        self.mu1 = nn.Parameter(torch.zeros(num_mixtures, num_dims))
        self.mu2 = nn.Parameter(torch.zeros(num_mixtures, num_dims))
        
        # Log Stds (std1, std2): M x D
        # Initialized to ones (log_std=0) as in reference loop (which makes softplus(1)~1.3)
        # We start with 0 (softplus(0)~0.7) to keep it well behaved, or 1 to match exactly.
        # Let's use 0.0 for stability.
        self.log_std1 = nn.Parameter(torch.zeros(num_mixtures, num_dims)) 
        self.log_std2 = nn.Parameter(torch.zeros(num_mixtures, num_dims))
        
        # Correlation (rho): M
        self.rho = nn.Parameter(torch.zeros(num_mixtures))

    @property
    def feature_dim(self):
        return self._feature_dim

    def get_rff_feature(self, x):
        """
        Compute RFF features for input x.
        x: (N, D)
        Vectorized across all mixtures for maximum performance.
        """
        N, D = x.shape
        M = self.num_mixtures
        S = self.rff_samples
        device = x.device

        # 1. Transform Parameters
        weights = F.softplus(self.raw_weights).view(M, 1, 1) # (M, 1, 1)
        std1 = F.softplus(self.log_std1).view(M, 1, D)    # (M, 1, D)
        std2 = F.softplus(self.log_std2).view(M, 1, D)    # (M, 1, D)
        mu1 = self.mu1.view(M, 1, D)                     # (M, 1, D)
        mu2 = self.mu2.view(M, 1, D)                     # (M, 1, D)
        rho = self.rho.view(M, 1, 1)                     # (M, 1, 1)

        # 2. Sample Frequencies (All mixtures at once)
        # eps: (M, S, D)
        eps1 = torch.randn(M, S, D, device=device)
        eps2 = torch.randn(M, S, D, device=device)

        # omega1: (M, S, D)
        omega1 = mu1 + std1 * eps1
        
        # omega2 (Conditional Bivariate Gaussian)
        # Using reference logic (std1 in noise term)
        term_mean = mu2 + rho * (std2 / std1) * (omega1 - mu1)
        term_noise = (1 - rho**2).sqrt() * std1 * eps2
        omega2 = term_mean + term_noise

        # 3. Compute inner products
        # x_spectral: (N, D) @ (M, D, S) -> (M, N, S)
        # We use matmul with expansion: (1, N, D) @ (M, D, S) -> (M, N, S)
        x_spectral1 = torch.matmul(x.unsqueeze(0), omega1.transpose(-1, -2)) * (2 * np.pi)
        x_spectral2 = torch.matmul(x.unsqueeze(0), omega2.transpose(-1, -2)) * (2 * np.pi)

        # 4. Feature Construction
        # scale: (M, 1, 1)
        scale = (weights / (4 * S)).sqrt()
        
        # cos part: (M, N, S)
        z_cos = scale * (x_spectral1.cos() + x_spectral2.cos())
        # sin part: (M, N, S) - Reference logic uses 2*sin1
        z_sin = scale * (x_spectral1.sin() + x_spectral1.sin())

        # Combine into (M, N, 2S)
        phi_mixtures = torch.cat([z_cos, z_sin], dim=-1)
        
        # Reshape and Reorder to (N, M, 2S) -> (N, M*2S)
        # (M, N, 2S) -> (N, M, 2S) -> (N, M*2S)
        return phi_mixtures.permute(1, 0, 2).reshape(N, -1)