import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleViewEncoder(nn.Module):
    """
    Encoder for a single view.
    Maps input view y_v to mu_v and log_sigma_v.
    """
    def __init__(self, input_dim, latent_dim, hidden_dims=[256, 128], arch_type='mlp', input_shape=None):
        """
        Args:
            input_dim: For vector inputs, this is the feature dimension.
                      For image inputs (cnn2d), this should be the total flattened size (C*H*W).
            latent_dim: Dimension of the latent code.
            hidden_dims: Hidden layer dimensions for MLP.
            arch_type: 'mlp', 'cnn' (1D), or 'cnn2d' (2D for images).
            input_shape: Tuple (C, H, W) for 'cnn2d' architecture. Auto-inferred if possible.
        """
        super().__init__()
        self.input_dim = input_dim
        self.arch_type = arch_type
        self.input_shape = input_shape
        
        if self.arch_type == 'mlp':
            layers = []
            curr_dim = input_dim
            for h_dim in hidden_dims:
                layers.append(nn.Linear(curr_dim, h_dim))
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(h_dim))
                curr_dim = h_dim
            self.trunk = nn.Sequential(*layers)
            
        elif self.arch_type == 'cnn':
            # 1D CNN for single view
            # Assumes input is (B, D) -> Unsqueeze to (B, 1, D)
            self.cnn = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2)
            )
            # Calculate output dim
            with torch.no_grad():
                dummy = torch.randn(1, 1, input_dim)
                out = self.cnn(dummy)
                self.cnn_out_dim = out.view(1, -1).shape[1]
            curr_dim = self.cnn_out_dim
            
        elif self.arch_type == 'cnn2d':
            # 2D CNN for image inputs (C, H, W) - 4 Layer Architecture
            if input_shape is None:
                raise ValueError("input_shape (C, H, W) must be provided for cnn2d architecture")
            
            C, H, W = input_shape
            self.cnn2d = nn.Sequential(
                # Layer 1: 3 -> 32
                nn.Conv2d(C, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                # Layer 2: 32 -> 64
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                # Layer 3: 64 -> 128
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                # Layer 4: 128 -> 256
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4))  # Fixed spatial size -> (B, 256, 4, 4)
            )
            # Calculate output dim
            with torch.no_grad():
                dummy = torch.randn(1, C, H, W)
                out = self.cnn2d(dummy)
                self.cnn2d_out_dim = out.view(1, -1).shape[1]
            curr_dim = self.cnn2d_out_dim
            
        else:
            raise ValueError(f"Unknown arch_type: {arch_type}")
            
        self.fc_mu = nn.Linear(curr_dim, latent_dim)
        self.fc_logvar = nn.Linear(curr_dim, latent_dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor.
               - For MLP/CNN1D: (B, D) vector
               - For CNN2D: (B, C, H, W) image tensor
        """
        if self.arch_type == 'mlp':
            # Expect (B, D)
            if x.dim() > 2:
                # If accidentally received image, flatten it
                x = x.view(x.size(0), -1)
            h = self.trunk(x)
            
        elif self.arch_type == 'cnn':
            # Expect (B, D), convert to (B, 1, D)
            if x.dim() == 2:
                x_in = x.unsqueeze(1)
            else:
                x_in = x
            h = self.cnn(x_in)
            h = h.view(h.size(0), -1)
            
        elif self.arch_type == 'cnn2d':
            # Expect (B, C, H, W)
            if x.dim() == 2:
                # Reshape flattened input back to image
                C, H, W = self.input_shape
                x = x.view(x.size(0), C, H, W)
            h = self.cnn2d(x)
            h = h.view(h.size(0), -1)
            
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class MultiViewEncoder(nn.Module):
    """
    Product-of-Experts Amortized Inference Network.
    
    Architecture:
    1. Independent encoders for each view q(z|y_v).
    2. Aggregation via Product-of-Experts (PoE) including a prior p(z) = N(0, I).
       q(z|Y) propto p(z) * prod_v q(z|y_v)
       
    This handles variable number of views naturally and is more robust.
    """
    def __init__(self, view_dims, latent_dim, hidden_dims=[256, 128], arch_type='mlp', view_shapes=None):
        """
        Args:
            view_dims (dict): {view_index/name: input_dim}
                             For vector views: input_dim is the feature count
                             For image views: input_dim is C*H*W (flattened size)
            latent_dim (int): Dimension of z
            view_shapes (dict, optional): {view_name: (C, H, W)} for image views using cnn2d
        """
        super().__init__()
        self.view_dims = view_dims
        self.latent_dim = latent_dim
        self.view_shapes = view_shapes or {}
        
        # Create an encoder for each view
        self.encoders = nn.ModuleDict()
        for v_name, v_dim in view_dims.items():
            # Convert key to string for ModuleDict
            key = str(v_name)
            input_shape = self.view_shapes.get(v_name, None)
            self.encoders[key] = SingleViewEncoder(
                input_dim=v_dim,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                arch_type=arch_type,
                input_shape=input_shape
            )
            
    def forward(self, views_dict):
        """
        Args:
            views_dict (dict): {view_name: tensor(B, D_v)}
            
        Returns:
            mu_joint, log_sigma_joint
        """
        # We will accumulate precision-weighted means and precisions
        # Prior expert: mu=0, var=1 => precision=1, weighted_mu=0
        
        # Initialize joint precision (T) and weighted mean (mu * T) with Prior
        # T_joint = I + sum(T_v)
        # mu_T_joint = 0 + sum(mu_v * T_v)
        
        # Get batch size from first available view
        first_view = next(iter(views_dict.values()))
        batch_size = first_view.size(0)
        device = first_view.device
        
        # Prior parameters
        mu_T_joint = torch.zeros(batch_size, self.latent_dim, device=device)
        T_joint = torch.ones(batch_size, self.latent_dim, device=device)
        
        for v_name, x_v in views_dict.items():
            key = str(v_name)
            if key not in self.encoders:
                continue
                
            # Encoding q(z|y_v)
            mu_v, logvar_v = self.encoders[key](x_v)
            # Tighter clamp: var in [1e-3, 10] to prevent both underflow and overflow
            logvar_v = logvar_v.clamp(min=-6.91, max=2.3) # exp(-6.91)≈1e-3, exp(2.3)≈10
            var_v = torch.exp(logvar_v) + 1e-6
            
            T_v = 1.0 / var_v  # Precision          
            # Aggregate (Product of Experts)
            mu_T_joint = mu_T_joint + mu_v * T_v
            T_joint = T_joint + T_v
                    # Joint Precision T = I + sum(T_v) (prior is N(0, I) -> precision I)
        # Add small jitter to inversion for stability
        sigma_joint = 1.0 / (T_joint + 1e-6)
        mu_joint = mu_T_joint * sigma_joint
        
        # Return mu and log_sigma
        log_sigma_joint = torch.log(torch.sqrt(sigma_joint))
        
        return mu_joint, log_sigma_joint
