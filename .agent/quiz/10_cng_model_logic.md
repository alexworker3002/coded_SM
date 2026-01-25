# CNG-MV-GPLVM 模型架构与代码逻辑详解

本文档详细介绍了 `src/models/cng_model.py` 中的代码逻辑、类结构以及涉及的数学原理。

## 1. 核心类定义：`CNG_MV_GPLVM`

`CNG_MV_GPLVM` 是整个框架的核心类，继承自 `nn.Module`。它整合了**变分推断 (VI)**、**线性纠错码 (ECC)** 以及**下一代频谱混合核 (NG-SM)**。

```python
class CNG_MV_GPLVM(nn.Module):
    """
    Coded Next-Gen Multi-View GPLVM (CNG-MV-GPLVM)
    
    该模型整合了:
    1. Variational Inference for Latents Z
    2. Linear Error Correcting Codes (ECC) for structural regularization
    3. Next-Gen Spectral Mixture Kernel (NG-SM) with RFF approximation
    """
```

### 1.1 初始化逻辑 `__init__`

初始化函数根据 `inference_mode` 参数决定了潜变量的建模方式。

```python
    def __init__(self, 
                 num_data,        # N: 样本总数
                 input_dim,       # d_z: 潜变量 Z 的维度
                 view_dims,       # Dict[str, int]: 各视图的名称和维度
                 redundancy_factor=2, # L: 编码冗余度
                 use_ecc=True,    # Switch: 是否启用 ECC
                 # ... 其他参数
                 inference_mode='direct', # 'direct' (GPLVM) or 'amortized' (VAE)
                 encoder_type='mlp' # 'mlp' or 'cnn'
                 ):
        # ...
        # 1. 变分推断策略
        if self.inference_mode == 'direct':
            self.q_mu = nn.Parameter(torch.randn(num_data, input_dim) * z_init_std)
            self.q_log_sigma = nn.Parameter(torch.ones(num_data, input_dim) * np.log(z_init_std))
        elif self.inference_mode == 'amortized':
            from src.models.components.encoder import MultiViewEncoder
            self.encoder = MultiViewEncoder(view_dims, input_dim, arch_type=encoder_type)
        elif self.inference_mode == 'semi_amortized':
            self.q_mu = nn.Parameter(torch.randn(num_data, input_dim) * z_init_std)
            self.q_log_sigma = nn.Parameter(torch.ones(num_data, input_dim) * np.log(z_init_std))
            from src.models.components.encoder import MultiViewEncoder
            self.encoder = MultiViewEncoder(view_dims, input_dim, arch_type=encoder_type)
```

**数学逻辑**：
*   **Direct (GPLVM)**：将潜变量 $Z$ 视为自由参数进行优化，不依赖输入 $X$。
*   **Amortized (VAE)**：通过编码器 $f_\phi(X)$ 预测 $Z$，即 $q(Z|X)$。
*   **Semi-Amortized**：混合模式，同时保留自由参数 $\mathbf{Z}_{opt}$ 和预测网络。

---

## 2. 潜变量获取与重采样

### 2.1 重参数化技巧 `reparameterize`

为了使梯度能够通过随机采样传递，使用了变分自编码器中经典的重参数化技巧：
$$ z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I) $$

```python
    def reparameterize(self, mu, log_sigma):
        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return mu + eps * std
```

### 2.2 潜变量路由 `get_latents`

该方法封装了多模式下的潜变量提取逻辑。

```python
    def get_latents(self, batch_indices=None, views_batch=None):
        if self.inference_mode == 'direct':
            return self.q_mu[batch_indices], self.q_log_sigma[batch_indices]
        elif self.inference_mode == 'amortized':
            return self.encoder(views_batch)
        elif self.inference_mode == 'semi_amortized':
            # 返回 (优化参数) 和 (编码器预测) 的元组
            mu_opt, log_var_opt = self.q_mu[batch_indices], self.q_log_sigma[batch_indices]
            mu_enc, log_sigma_enc = self.encoder(views_batch)
            return (mu_opt, log_var_opt), (mu_enc, log_sigma_enc)
```

---

## 3. 前向传播与 ECC 编码

`forward` 方法实现了从潜空间 $Z$ 到观测空间 $Y$ 的端到端映射。

```python
    def forward(self, batch_indices=None, views_batch=None):
        # 1. 采样 Z (处理 Semi 模式的特殊返回)
        latents = self.get_latents(batch_indices, views_batch)
        if self.inference_mode == 'semi_amortized':
            (mu_opt, log_sigma_opt), (mu_enc, _) = latents
            z_sample = self.reparameterize(mu_opt, log_sigma_opt)
            batch_mu, batch_log_sigma = mu_opt, log_sigma_opt
        else:
            batch_mu, batch_log_sigma = latents
            z_sample = self.reparameterize(batch_mu, batch_log_sigma)
            mu_enc = None

        # 2. ECC 编码: Z -> X
        # 通过线性投影矩阵 G，将 Z 映射到具有冗余度的空间 X = ZG
        x_sample = self.ecc_module(z_sample)
        
        # 3. 多视图映射 (通过 NG-SM 核函数的 RFF 近似)
        y_recons = {}
        for name in self.view_dims.keys():
            # X -> Phi(X) (RFF 特征提取)
            features = self.kernels[name].get_rff_feature(x_sample)
            # Phi -> Y (线性 Readout 层)
            y_recons[name] = self.readouts[name](features)
```

**数学逻辑**：
*   **ECC 编码**：引入 $L$ 倍冗余。对于 $L=2$ 的重复码，映射为 $[Z, Z]$；对于随机高斯码，映射为 $Z \cdot \mathbf{G}_{rand}$。这增强了模型在视图丢失时的几何稳定性。
*   **NG-SM Kernel**：利用随机傅里叶特征 (RFF) 将非线性核函数计算转化为高维空间的线性回归，大幅提升了多视图扩展性。

---

## 4. 损失函数计算 `compute_loss`

该方法计算变分下界 (ELBO) 的负值，用于梯度下降。

```python
    def compute_loss(self, views_batch, batch_indices, beta=1.0):
        outputs = self.forward(batch_indices, views_batch)
        # 解包 (处理模式兼容性)
        if self.inference_mode == 'semi_amortized':
            y_recons, mu, log_sigma, mu_enc = outputs
        else:
            y_recons, mu, log_sigma = outputs

        # --- 1. 重建损失 (Reconstruction Loss) ---
        # 假设各视图服从高斯分布，计算 NLL
        total_recon_loss = 0.0
        for name, y_true in views_batch.items():
            y_pred = y_recons[name]
            noise_sigma = torch.exp(self.log_noise_sigmas[name])
            mse = (y_true - y_pred).pow(2)
            nll = torch.log(noise_sigma) + 0.5 * mse / (noise_sigma ** 2)
            total_recon_loss += nll.sum()

        # --- 2. KL 散度 (KL Divergence) ---
        # 约束近似后验 q(Z) 接近标准正态分布 p(Z) = N(0, I)
        var = torch.exp(2 * log_sigma)
        kl_div = -0.5 * torch.sum(1 + 2 * log_sigma - mu.pow(2) - var)
        
        return total_recon_loss + beta * kl_div, details
```

**数学逻辑**：
*   **负对数似然 (NLL)**：衡量模型对原始视图的重建精度。
*   **对齐损失 (Alignment Loss)**：虽然 `compute_loss` 本身只返回基础 ELBO，但在 `engine.py` 中，如果处于 `semi_amortized` 模式，会额外计算 $\| \mu_{opt} - \mu_{enc} \|^2$，强迫编码器拟合 SMLVM 的最优潜空间路径。
