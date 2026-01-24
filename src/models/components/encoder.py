import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiViewEncoder(nn.Module):
    """
    Amortized Inference Network for Multi-View Data.
    q(z|y) = N(mu(y), sigma(y))
    
    Inputs:
        views_dict: {view_name: tensor(Batch, Dim)}
    """
    def __init__(self, view_dims, latent_dim, hidden_dims=[256, 128], arch_type='mlp'):
        super().__init__()
        self.view_dims = view_dims
        self.latent_dim = latent_dim
        self.arch_type = arch_type
        
        # Calculate total input dimension
        self.input_dim = sum(view_dims.values())
        
        if self.arch_type == 'mlp':
            # Build MLP
            layers = []
            curr_dim = self.input_dim
            
            for h_dim in hidden_dims:
                layers.append(nn.Linear(curr_dim, h_dim))
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(h_dim))
                curr_dim = h_dim
                
            self.shared_trunk = nn.Sequential(*layers)
            
        elif self.arch_type == 'cnn':
            # Build 1D CNN
            # Assumption: Input is concatenated feature vector.
            # We treat it as 1 channel signal: (B, 1, InputDim)
            # Simple Architecture: Conv -> Pool -> Conv -> Pool -> Flatten
            
            self.cnn = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(2)
            )
            
            # Compute output dim
            # L_out = floor((L_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
            # Pool: L_out = L_in / 2
            # Let's compute manually or dry run.
            
            # Rough calc: InputDim -> /2 -> /2 = InputDim / 4
            # We will use a linear layer to map flattened output to last hidden dim
            # to match the head structure if possible, or just go directly to heads.
            
            with torch.no_grad():
                dummy = torch.randn(1, 1, self.input_dim)
                out = self.cnn(dummy)
                self.cnn_out_dim = out.view(1, -1).shape[1]
                
            curr_dim = self.cnn_out_dim

        else:
            raise ValueError(f"Unknown arch_type: {arch_type}")
        
        # Heads for Mu and LogSigma
        self.fc_mu = nn.Linear(curr_dim, latent_dim)
        self.fc_logvar = nn.Linear(curr_dim, latent_dim)
        
    def forward(self, views_dict):
        # 1. Concatenate all views (ensure deterministic order)
        # Sort keys to guarantee consistent order
        sorted_keys = sorted(self.view_dims.keys())
        concat_input = torch.cat([views_dict[k] for k in sorted_keys], dim=1)
        
        # 2. Shared representation
        if self.arch_type == 'mlp':
            hidden = self.shared_trunk(concat_input)
        elif self.arch_type == 'cnn':
            # Reshape for Conv1d: (B, 1, D)
            x = concat_input.unsqueeze(1)
            feat = self.cnn(x)
            hidden = feat.view(feat.size(0), -1) # Flatten
        
        # 3. Predict parameters
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        
        # Return mu and log_sigma (logvar = 2 * log_sigma)
        # So log_sigma = 0.5 * logvar
        log_sigma = 0.5 * logvar
        
        return mu, log_sigma
