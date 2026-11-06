"""Fixed real-valued redundant projections shared by generation and decoding."""

import pickle
from pathlib import Path

import torch
from torch import nn


class LinearECCProjection(nn.Module):
    """Map row-vector latents to redundant coordinates: ``X = Z @ G``."""

    def __init__(self, z_dim, redundancy_factor, mode="repetition", matrix_path=None):
        super().__init__()
        if z_dim < 1 or redundancy_factor < 1 or int(redundancy_factor) != redundancy_factor:
            raise ValueError("z_dim and redundancy_factor must be positive integers")
        self.z_dim = z_dim
        self.redundancy_factor = int(redundancy_factor)
        self.x_dim = z_dim * self.redundancy_factor
        if matrix_path is not None:
            with Path(matrix_path).open("rb") as handle:
                matrix = torch.as_tensor(pickle.load(handle), dtype=torch.float32)
            if matrix.shape != (z_dim, self.x_dim) and matrix.T.shape == (z_dim, self.x_dim):
                matrix = matrix.T
        elif mode == "random_gaussian":
            matrix = torch.randn(z_dim, self.x_dim) / self.x_dim**0.5
        elif mode == "repetition":
            matrix = torch.kron(torch.eye(z_dim), torch.ones(1, self.redundancy_factor))
        else:
            raise ValueError(f"Unknown ECC mode: {mode}")
        if matrix.shape != (z_dim, self.x_dim) or not torch.isfinite(matrix).all():
            raise ValueError("Invalid ECC matrix shape or non-finite entries")
        if torch.linalg.matrix_rank(matrix) != z_dim:
            raise ValueError("ECC matrix must have full row rank")
        self.register_buffer("G", matrix.contiguous())

    def forward(self, z):
        with torch.amp.autocast(device_type=z.device.type, enabled=False):
            return z.float() @ self.G.float()
