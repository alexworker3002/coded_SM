import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle

class LinearECCDecoder(nn.Module):
    """
    Linear ECC Decoder (Decoding Layer) for Coded-Amortized Inference.
    
    Function:
    Inverts the linear code X = Z @ G to recover Z from a predicted distribution q(X|Y).
    
    Math:
    Let X ~ N(mu_X, diag(var_X)).
    We estimate Z using the pseudo-inverse projection G_pinv = (G^T G)^-1 G^T = G^dag.
    (Note: Since Z is row vector 1xK, X = ZG, so Z = X G^dag)
    
    mu_Z = mu_X @ G_pinv.T
    var_Z_j = sum_i (G_pinv_ji^2 * var_X_i)  [Mean Field Approximation]
    """
    def __init__(self, z_dim, redundancy_factor, mode='repetition', matrix_path=None):
        super().__init__()
        self.z_dim = z_dim
        self.redundancy_factor = redundancy_factor
        self.x_dim = z_dim * redundancy_factor
        
        # Load or Generate G (Must match the Encoder/Generator logic exactly)
        G = self._load_generation_matrix(matrix_path, z_dim, self.x_dim, mode)
        
        # Compute Pseudo-Inverse G_pinv
        # G shape: (z_dim, x_dim)
        # G_pinv shape: (x_dim, z_dim) such that G @ G_pinv ~ I (if full rank)
        # Wait, standard def: A x = b. Here Z G = X. 
        # So Z = X G^T (G G^T)^-1 ? No.
        # Let's use torch.pinverse which handles (N, M) -> (M, N)
        # If G: (Z, X), then G_pinv: (X, Z).
        # Z (1, Z) = X (1, X) @ G_pinv (X, Z). Correct.
        
        G_pinv = torch.pinverse(G)
        
        # Register buffers
        self.register_buffer('G', G)
        self.register_buffer('G_pinv', G_pinv)
        self.register_buffer('G_sq', G_pinv.pow(2)) # For variance propagation
        
        print(f"[ECC Decoder] Initialized {mode} decoder. X({self.x_dim}) -> Z({z_dim})")

    def _load_generation_matrix(self, path, z_dim, x_dim, mode):
        """
        Duplicate logic from LinearECCProjection to ensure G is identical.
         ideally this should be shared code, but for now copying is safer than refactoring.
        """
        if path and os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    G_numpy = pickle.load(f)
                G_tensor = torch.tensor(G_numpy, dtype=torch.float32)
                if G_tensor.shape != (z_dim, x_dim):
                     if G_tensor.T.shape == (z_dim, x_dim):
                         G_tensor = G_tensor.T
                return G_tensor
            except Exception:
                pass

        if mode == 'random_gaussian':
            std = 1.0 / (x_dim ** 0.5)
            # Use fixed seed for deterministic G if possible, but here we assume the 
            # model loading state_dict will handle synchronization if trained.
            # For fresh init, we trust randomness or seed.
            return torch.randn(z_dim, x_dim) * std
            
        else: # repetition
            repeats = x_dim // z_dim
            eye = torch.eye(z_dim)
            ones = torch.ones(1, repeats)
            return torch.kron(eye, ones)

    def forward(self, mu_x, logvar_x):
        """
        Args:
            mu_x: (Batch, x_dim)
            logvar_x: (Batch, x_dim)
            
        Returns:
            mu_z: (Batch, z_dim)
            logvar_z: (Batch, z_dim)
        """
        # 1. Decode Mean: mu_z = mu_x @ G_pinv
        # G_pinv is (X, Z). mu_x is (B, X). Result (B, Z).
        mu_z = mu_x @ self.G_pinv
        
        # 2. Decode Variance: var_z = var_x @ G_pinv^2
        var_x = torch.exp(logvar_x)
        var_z = var_x @ self.G_sq
        
        # Log space
        logvar_z = torch.log(var_z + 1e-6)
        
        return mu_z, logvar_z
