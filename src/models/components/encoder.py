import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleViewEncoder(nn.Module):
    """
    Encoder for a single view.
    Maps input view y_v to mu_v and log_sigma_v.
    """
    def __init__(self, input_dim, latent_dim, hidden_dims=[256, 128], arch_type='mlp'):
        super().__init__()
        self.input_dim = input_dim
        self.arch_type = arch_type
        
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
            
        else:
            raise ValueError(f"Unknown arch_type: {arch_type}")
            
        self.fc_mu = nn.Linear(curr_dim, latent_dim)
        self.fc_logvar = nn.Linear(curr_dim, latent_dim)

    def forward(self, x):
        if self.arch_type == 'mlp':
            h = self.trunk(x)
        elif self.arch_type == 'cnn':
            x_in = x.unsqueeze(1) # B, 1, D
            h = self.cnn(x_in)
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
    def __init__(self, view_dims, latent_dim, hidden_dims=[256, 128], arch_type='mlp'):
        """
        Args:
            view_dims (dict): {view_index/name: input_dim}
            latent_dim (int): Dimension of z
        """
        super().__init__()
        self.view_dims = view_dims
        self.latent_dim = latent_dim
        
        # Create an encoder for each view
        self.encoders = nn.ModuleDict()
        for v_name, v_dim in view_dims.items():
            # Convert key to string for ModuleDict
            key = str(v_name)
            self.encoders[key] = SingleViewEncoder(
                input_dim=v_dim,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                arch_type=arch_type
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
            var_v = torch.exp(logvar_v) + 1e-6 # Stability
            T_v = 1.0 / var_v
            
            # Aggregate (Product of Experts)
            mu_T_joint = mu_T_joint + mu_v * T_v
            T_joint = T_joint + T_v
            
        # Compute joint posterior parameters
        # Sigma = 1 / T
        sigma_joint = 1.0 / T_joint
        mu_joint = mu_T_joint * sigma_joint
        
        # Return mu and log_sigma
        log_sigma_joint = torch.log(torch.sqrt(sigma_joint))
        
        return mu_joint, log_sigma_joint
