"""Decode means and diagonal variances using the generator's actual matrix."""

import torch
from torch import nn


class LinearECCDecoder(nn.Module):
    """Decode X = ZG; inputs and outputs both use log standard deviation.

    The returned variance is the diagonal of the transformed covariance.
    Off-diagonal covariance is intentionally omitted by this mean-field model.
    """

    def __init__(self, generation_matrix):
        super().__init__()
        matrix = torch.as_tensor(generation_matrix).detach().clone()
        if matrix.ndim != 2 or not torch.isfinite(matrix).all():
            raise ValueError("Expected a finite two-dimensional generation matrix")
        if torch.linalg.matrix_rank(matrix) != matrix.shape[0]:
            raise ValueError("Generation matrix must have full row rank")
        inverse = torch.linalg.pinv(matrix)
        self.register_buffer("G", matrix)
        self.register_buffer("G_pinv", inverse)
        self.register_buffer("G_sq", inverse.square())

    def forward(self, mu_x, log_sigma_x):
        with torch.amp.autocast(device_type=mu_x.device.type, enabled=False):
            mu_z = mu_x.float() @ self.G_pinv.float()
            var_z = torch.exp(2 * log_sigma_x.float()) @ self.G_sq.float()
            log_sigma_z = 0.5 * var_z.clamp_min(torch.finfo(var_z.dtype).tiny).log()
        return mu_z, log_sigma_z
