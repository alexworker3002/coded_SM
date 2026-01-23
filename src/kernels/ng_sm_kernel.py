# src/kernels/ng_sm_kernel.py

import torch
import math
import gpytorch
from gpytorch.kernels import Kernel

class NextGenSpectralMixtureKernel(Kernel):
    """
    Next-Gen Spectral Mixture Kernel (NG-SM)
    基于 Yang et al. (2025) 的实现。
    使用双变量高斯混合模型拟合谱密度，并利用 RFF 进行近似。
    """
    def __init__(self, num_dims, num_mixtures=4, rff_samples=500, **kwargs):
        super().__init__(**kwargs)
        self.num_dims = num_dims
        self.num_mixtures = num_mixtures # Q
        self.rff_samples = rff_samples   # S
        
        # 1. 注册原始参数 (Raw Parameters)
        # 我们让 GPyTorch 帮我们要处理约束 (Constraints)，比如方差必须大于0
        self.register_parameter(
            name="raw_mixture_weights", 
            parameter=torch.nn.Parameter(torch.zeros(self.num_mixtures))
        )
        self.register_parameter(
            name="raw_mixture_means", 
            parameter=torch.nn.Parameter(torch.zeros(self.num_mixtures, self.num_dims))
        )
        self.register_parameter(
            name="raw_mixture_scales", 
            parameter=torch.nn.Parameter(torch.zeros(self.num_mixtures, self.num_dims))
        )

        # 2. 注册约束 (Constraints)
        # 权重和尺度必须为正，使用 Softplus 变换
        self.register_constraint("raw_mixture_weights", gpytorch.constraints.Positive())
        self.register_constraint("raw_mixture_scales", gpytorch.constraints.Positive())
        
        # RFF 采样所需的随机相位和频率权重 (固定不训练)
        # 注意：这里我们先占位，实际 forward 时再根据设备生成或缓存
        self.register_buffer("random_weights", torch.randn(self.num_mixtures, self.rff_samples, self.num_dims))
        # REMOVED: self.register_buffer("random_biases", ...) - Using Cos/Sin concatenation instead

    @property
    def mixture_weights(self):
        return self.raw_mixture_weights_constraint.transform(self.raw_mixture_weights)

    @property
    def mixture_scales(self):
        return self.raw_mixture_scales_constraint.transform(self.raw_mixture_scales)

    @property
    def mixture_means(self):
        return self.raw_mixture_means

    def forward(self, x1, x2, diag=False, **params):
        """
        NG-SM 通常配合 RFF 使用，不直接计算 Gram 矩阵。
        但在标准 GPyTorch 框架下，如果要兼容 exact inference，仍需提供 standard forward。
        
        这里我们主要实现 RFF 特征映射逻辑供 VariationalStrategy 调用。
        """
        # 这是一个占位符，因为我们的核心是用 get_rff_feature
        # 如果你只做变分推断 + RFF，这个 standard forward 其实不会被大规模调用
        raise NotImplementedError("NG-SM Kernel intended for RFF use only in this project.")

    def get_rff_feature(self, x):
        """
        计算随机傅里叶特征 Z(x)
        x: [Batch, D]
        Returns: [Batch, 2 * Q * S] (实部和虚部拼接)
        """
        # 1. 获取变换后的非负参数
        w = self.mixture_weights  # [Q]
        m = self.mixture_means    # [Q, D]
        s = self.mixture_scales   # [Q, D]
        
        # 2. 这里的数学逻辑需严格参考 NG-MVLVM 的 param_gp.py
        # phi(x) = [ sqrt(w/S) * cos( (s * rand_n + m) * x ) , sqrt(w/S) * sin( (...) * x ) ]
        # 注意维度的广播 (Broadcasting)
        
        # 扩展维度以进行广播
        # x: [Batch, 1, 1, D]
        x_expanded = x.unsqueeze(1).unsqueeze(1) 
        
        # random_weights: [Q, S, D]
        # spectral_freqs ~ N(m, s^2) => s * rand_n + m
        spectral_freqs = s.unsqueeze(1) * self.random_weights + m.unsqueeze(1) # [Q, S, D]
        
        # 计算内积: sum_d (freq_d * x_d)
        # [Batch, Q, S]
        inner_prod = torch.sum(spectral_freqs.unsqueeze(0) * x_expanded, dim=-1)
        
        # 计算特征
        # scaling factor: sqrt(w / S)
        # S is self.rff_samples
        amplitude = torch.sqrt(w.view(1, -1, 1) / self.rff_samples)
        
        z_cos = amplitude * torch.cos(inner_prod)
        z_sin = amplitude * torch.sin(inner_prod)
        
        # 拼接并展平: [Batch, 2 * Q * S]
        # cat dim=-1 makes it [Batch, Q, 2*S] -> view -> [Batch, 2*Q*S]
        feature = torch.cat([z_cos, z_sin], dim=-1).view(x.size(0), -1)
        return feature