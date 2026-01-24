# src/modules/ecc.py

import torch
import torch.nn as nn
import pickle
import os

class LinearECCProjection(nn.Module):
    """
    线性纠错编码层 (Linear Error Correcting Code Projection)
    将低维潜变量 Z 映射到高维冗余空间 X。
    公式: X = Z @ G
    """
    def __init__(self, z_dim, redundancy_factor, mode='repetition', matrix_path=None):
        super().__init__()
        self.z_dim = z_dim
        self.redundancy_factor = redundancy_factor
        self.x_dim = z_dim * redundancy_factor
        
        # 加载生成矩阵 G
        G = self._load_generation_matrix(matrix_path, z_dim, self.x_dim, mode)
        
        # 关键点：将 G 注册为 buffer
        self.register_buffer('G', G)

    def _load_generation_matrix(self, path, z_dim, x_dim, mode):
        """
        加载或生成矩阵
        mode: 'repetition' | 'random_gaussian'
        """
        if path and os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    G_numpy = pickle.load(f)
                G_tensor = torch.tensor(G_numpy, dtype=torch.float32)
                if G_tensor.shape != (z_dim, x_dim):
                    if G_tensor.T.shape == (z_dim, x_dim):
                        G_tensor = G_tensor.T
                    else:
                        raise ValueError(f"Matrix shape mismatch: {G_tensor.shape}")
                print(f"✅ Loaded ECC Matrix from {path}")
                return G_tensor
            except Exception as e:
                print(f"⚠️ Failed to load matrix: {e}. Falling back to generation mode: {mode}")

        # Generation Logic
        if mode == 'random_gaussian':
            # Path A: Random Gaussian Code (Johnson-Lindenstrauss)
            # Entries ~ N(0, 1/d_x) to preserve expected norm ||x|| = ||z||?
            # Or N(0, 1/L)? Check Math Doc.
            # Doc says: G_ij ~ N(0, 1 / (L * d_z)) to preserve TOTAL norm.
            # std = 1.0 / sqrt(x_dim)
            std = 1.0 / (x_dim ** 0.5)
            print(f"ℹ️ Generating Random Gaussian Matrix (std={std:.4f})")
            return torch.randn(z_dim, x_dim) * std
            
        else: # mode == 'repetition'
            # Path B: Block Repetition (Kronecker I (x) 1)
            print(f"ℹ️ Generating Repetition Code Matrix (L={self.redundancy_factor})")
            repeats = x_dim // z_dim
            eye = torch.eye(z_dim)
            ones = torch.ones(1, repeats)
            return torch.kron(eye, ones)

    def forward(self, z):
        """
        输入: z [Batch, z_dim]
        输出: x [Batch, x_dim]
        """
        # 矩阵乘法
        return z @ self.G