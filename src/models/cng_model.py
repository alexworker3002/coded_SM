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
                 ecc_matrix_path=None, # 预定义的 ECC 矩阵路径
                 inference_mode='direct', # 'direct' (GPLVM) or 'amortized' (VAE)
                 ecc_mode='repetition', # 'repetition' or 'random_gaussian'
                 encoder_type='mlp' # 'mlp' or 'cnn'
                 ):
        super().__init__()
        
        self.num_data = num_data
        self.input_dim = input_dim
        self.view_dims = view_dims
        self.use_ecc = use_ecc
        self.redundancy_factor = redundancy_factor
        self.inference_mode = inference_mode
        self.ecc_mode = ecc_mode
        self.encoder_type = encoder_type
        
        # =========================================================
        # 1. Variational Inference Strategy
        # =========================================================
        if self.inference_mode == 'direct':
            print("[Model] Using Direct Optimization Inference (Standard GPLVM)")
            self.q_mu = nn.Parameter(torch.randn(num_data, input_dim) * z_init_std)
            self.q_log_sigma = nn.Parameter(torch.ones(num_data, input_dim) * np.log(z_init_std))
        elif self.inference_mode == 'amortized':
            print(f"[Model] Using Amortized Inference (Deep Encoder: {encoder_type})")
            from src.models.components.encoder import MultiViewEncoder
            self.encoder = MultiViewEncoder(view_dims, input_dim, arch_type=encoder_type)
        elif self.inference_mode == 'semi_amortized':
            print(f"[Model] Using SEMI-Amortized Inference (Direct + Encoder: {encoder_type})")
            # 1. Direct Parameters (for Z_opt)
            self.q_mu = nn.Parameter(torch.randn(num_data, input_dim) * z_init_std)
            self.q_log_sigma = nn.Parameter(torch.ones(num_data, input_dim) * np.log(z_init_std))
            # 2. Encoder (to be aligned)
            from src.models.components.encoder import MultiViewEncoder
            self.encoder = MultiViewEncoder(view_dims, input_dim, arch_type=encoder_type)
        else:
            raise ValueError(f"Unknown inference mode: {inference_mode}")
        
        # =========================================================
        # 2. ECC Module (Optional)
        # =========================================================
        if self.use_ecc:
            self.x_dim = input_dim * redundancy_factor
            self.ecc_module = LinearECCProjection(
                input_dim, 
                redundancy_factor, 
                mode=ecc_mode, 
                matrix_path=ecc_matrix_path
            )
            print(f"[Model] Initialized with ECC ({ecc_mode}). Z({input_dim}) -> X({self.x_dim})")
        else:
            self.x_dim = input_dim
            self.ecc_module = nn.Identity()
            print(f"[Model] Initialized WITHOUT ECC (Baseline Mode). Z({input_dim}) -> X({self.x_dim})")

        # =========================================================
        # 3. Multi-View Components (Kernels + Readouts)
        # =========================================================
        self.kernels = nn.ModuleDict()
        self.readouts = nn.ModuleDict()
        self.log_noise_sigmas = nn.ParameterDict()
        
        # Feature dim will be obtained from kernel after initialization
        self.num_mixtures = num_mixtures
        self.rff_samples = rff_samples
        
        for name, v_dim in view_dims.items():
            # Kernel
            self.kernels[name] = NextGenSpectralMixtureKernel(
                num_dims=self.x_dim,
                num_mixtures=num_mixtures, 
                rff_samples=rff_samples
            )
            # Readout (use kernel's feature_dim property)
            self.readouts[name] = nn.Linear(self.kernels[name].feature_dim, v_dim, bias=True)
            # Noise (init log(-2) ~ 0.135)
            self.log_noise_sigmas[name] = nn.Parameter(torch.tensor(-2.0))

    def reparameterize(self, mu, log_sigma):
        """
        z = mu + sigma * epsilon
        """
        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def get_latents(self, batch_indices=None, views_batch=None):
        """
        根据推断模式获取 q(z) 参数
        Semi-Amortized: returns (mu_opt, log_sigma_opt) AND (mu_enc, log_sigma_enc)
        """
        if self.inference_mode == 'direct':
            if batch_indices is None:
                return self.q_mu, self.q_log_sigma
            else:
                return self.q_mu[batch_indices], self.q_log_sigma[batch_indices]
        
        elif self.inference_mode == 'amortized':
            if views_batch is None:
                raise ValueError("In amortized mode, views_batch must be provided to forward()")
            return self.encoder(views_batch)
            
        elif self.inference_mode == 'semi_amortized':
            # Direct Part
            if batch_indices is None:
                mu_opt, log_var_opt = self.q_mu, self.q_log_sigma
            else:
                mu_opt, log_var_opt = self.q_mu[batch_indices], self.q_log_sigma[batch_indices]
            
            # Encoder Part
            if views_batch is None:
                # During global eval or something, we might want just Opt?
                # But typically we provide batch.
                mu_enc, log_sigma_enc = None, None
            else:
                mu_enc, log_sigma_enc = self.encoder(views_batch)
                
            return (mu_opt, log_var_opt), (mu_enc, log_sigma_enc)

    def forward(self, batch_indices=None, views_batch=None):
        """
        前向传播
        """
        # 1. 采样 Z
        latents = self.get_latents(batch_indices, views_batch)
        
        if self.inference_mode == 'semi_amortized':
            (mu_opt, log_sigma_opt), (mu_enc, _) = latents
            # For reconstruction, we use Z_opt (Direct Optimization)
            z_sample = self.reparameterize(mu_opt, log_sigma_opt)
            # We return mu_enc for Alignment Loss calculation in engine
            batch_mu = mu_opt
            batch_log_sigma = log_sigma_opt
        else:
            batch_mu, batch_log_sigma = latents
            z_sample = self.reparameterize(batch_mu, batch_log_sigma)
            mu_enc = None # Placeholder
        
        # 2. ECC 编码: Z -> X
        x_sample = self.ecc_module(z_sample)
        
        # 3. Multi-View Mapping (RFF Features for GP Loss)
        y_recons = {}
        view_features = {}
        
        for name in self.view_dims.keys():
            # Kernel Mapping: X -> Phi_v(X)
            features = self.kernels[name].get_rff_feature(x_sample)
            view_features[name] = features
            
            # Readout: Phi_v -> Y_hat_v (For standard VAE loss or prediction)
            y_recons[name] = self.readouts[name](features)
        
        if self.inference_mode == 'semi_amortized':
            return y_recons, batch_mu, batch_log_sigma, mu_enc, view_features
            
        return y_recons, batch_mu, batch_log_sigma, view_features

    def _compute_gp_marginal_nll(self, Phi, y_true, noise_sigma):
        """
        利用 Woodbury 恒等式高效计算 GP 边际似然的负对数
        Phi: (N, D) - RFF 特征
        y_true: (N, Dy) - 观测数据
        noise_sigma: (Scalar) - 噪声标准差
        """
        N, D = Phi.shape
        Dy = y_true.shape[1]
        noise_var = noise_sigma ** 2
        jitter = 1e-6
        
        if N > D:
            # Woodbury Identity Case (Scalable)
            # A = Phi^T Phi + sigma^2 I
            A = Phi.t() @ Phi + (noise_var + jitter) * torch.eye(D, device=Phi.device)
            L = torch.linalg.cholesky(A)
            
            # Lt_inv_Phi_y = L^{-1} * (Phi^T * Y)
            Phi_T_Y = Phi.t() @ y_true
            L_inv_Phi_Y = torch.linalg.solve_triangular(L, Phi_T_Y, upper=False)
            
            # neg_log_lik = 0.5 * [ (Y^T Y - ||L_inv_Phi_Y||^2) / noise_var + 2*log|L| + (N-D)*log(noise_var) + N*log(2pi) ]
            # Note: We divide by N and Dy to keep scale stable for different datasets
            y_sq_sum = y_true.pow(2).sum()
            quad_term = (y_sq_sum - L_inv_Phi_Y.pow(2).sum()) / noise_var
            
            log_det_A = 2 * torch.log(torch.diag(L)).sum()
            # log|K| = log|Phi Phi^T + sigma^2 I| = log|Phi^T Phi + sigma^2 I| + (N-D)log(sigma^2)
            log_det_K = log_det_A + (N - D) * torch.log(torch.tensor(noise_var))
            
            nll = 0.5 * (quad_term + log_det_K * Dy + N * Dy * np.log(2 * np.pi))
        else:
            # Direct Case (N <= D)
            K = Phi @ Phi.t() + (noise_var + jitter) * torch.eye(N, device=Phi.device)
            L = torch.linalg.cholesky(K)
            L_inv_Y = torch.linalg.solve_triangular(L, y_true, upper=False)
            
            quad_term = L_inv_Y.pow(2).sum()
            log_det_K = 2 * torch.log(torch.diag(L)).sum()
            nll = 0.5 * (quad_term + log_det_K * Dy + N * Dy * np.log(2 * np.pi))
            
        return nll

    def compute_loss(self, views_batch, batch_indices, beta=1.0, use_gp_loss=True):
        """
        views_batch: dict {view_name: tensor}
        use_gp_loss: 是否使用 Yang (2025) 的 GP 边际似然损失
        """
        # Pass views_batch to forward for Amortized Inference support
        outputs = self.forward(batch_indices, views_batch)
        
        if self.inference_mode == 'semi_amortized':
            y_recons, mu, log_sigma, mu_enc, view_features = outputs
        else:
            y_recons, mu, log_sigma, view_features = outputs
            mu_enc = None
        
        total_data_loss = 0.0
        details = {}
        
        # --- 1. Data Loss (GP Marginal NLL or Reconstruction MSE) ---
        for name, y_true in views_batch.items():
            noise_sigma = torch.exp(self.log_noise_sigmas[name])
            
            if use_gp_loss:
                # Yang (2025) 核心逻辑: GP Marginal Likelihood
                Phi = view_features[name]
                view_loss = self._compute_gp_marginal_nll(Phi, y_true, noise_sigma)
            else:
                # 标准 VAE 逻辑: Gaussian Reconstruction NLL
                y_pred = y_recons[name]
                mse = (y_true - y_pred).pow(2)
                nll = torch.log(noise_sigma) + 0.5 * mse / (noise_sigma ** 2)
                view_loss = nll.sum()
            
            total_data_loss += view_loss
            details[f"recon_{name}"] = view_loss.item()
            details[f"sigma_{name}"] = noise_sigma.item()
        
        # --- 2. KL Divergence for Z (Yang 2025 Scaling: 1/(N*50)) ---
        var = torch.exp(2 * log_sigma)
        kl_div_raw = -0.5 * torch.sum(1 + 2 * log_sigma - mu.pow(2) - var)
        kl_div_scaled = kl_div_raw / (self.num_data * 50)  # Yang (2025) scaling
        
        details["kl_loss"] = kl_div_raw.item()
        details["kl_scaled"] = kl_div_scaled.item()
        details["data_loss"] = total_data_loss.item() 
        
        # Total loss = Data Loss + Scaled KL (beta is ignored when using Yang scaling)
        return total_data_loss + kl_div_scaled, details
