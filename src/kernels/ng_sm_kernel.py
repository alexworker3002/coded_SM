
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
        """
        # Parameters
        weights = F.softplus(self.raw_weights) # M x 1
        std1 = F.softplus(self.log_std1) # M x D
        std2 = F.softplus(self.log_std2) # M x D
        # rho is used directly.
        
        N = x.size(0)
        phi_list = []
        
        # Loop over mixtures (as in reference) for Bivariate Sampling
        for i in range(self.num_mixtures):
            device = x.device
            
            eps1 = torch.randn(self.rff_samples, self.num_dims, device=device)
            eps2 = torch.randn(self.rff_samples, self.num_dims, device=device)
            
            # Extract parameters for i-th mixture
            m1_i = self.mu1[i] # D
            m2_i = self.mu2[i] # D
            s1_i = std1[i] # D
            s2_i = std2[i] # D
            rho_i = self.rho[i] # Scalar
            
            # Step 1: Sample omega1
            omega1 = m1_i + s1_i * eps1 # S x D
            
            # Step 2: Sample omega2 (Conditional)
            # REPLICATING REFERENCE LOGIC (including potential anomaly std1 usage in noise term)
            # Reference:
            # sampled_spectral_pt2 = mu2 + rho * (std2/std1) * (omega1 - mu1) + sqrt(1-rho^2) * std1 * eps2
            
            term_mean = m2_i + rho_i * (s2_i / s1_i) * (omega1 - m1_i)
            # Using s1_i for noise term as in reference
            term_noise = torch.sqrt(1 - rho_i ** 2) * s1_i * eps2 
            
            omega2 = term_mean + term_noise # S x D
            
            # 2. Compute Features
            # x_spectral1 = 2pi * x @ omega1.T
            x_spectral1 = (2 * np.pi) * x @ omega1.t() # N x S
            x_spectral2 = (2 * np.pi) * x @ omega2.t() # N x S
            
            # Phi_i = sqrt(w / 4S) * [cos1+cos2, sin1+sin1]
            scale = torch.sqrt(weights[i] / (4 * self.rff_samples))
            
            z_cos = x_spectral1.cos() + x_spectral2.cos()
            z_sin = x_spectral1.sin() + x_spectral1.sin() # Reference logic
            
            phi_i = scale * torch.cat([z_cos, z_sin], dim=1) # N x 2S
            phi_list.append(phi_i)
            
        return torch.cat(phi_list, dim=1) # N x (M * 2S)