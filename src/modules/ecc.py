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
    def __init__(self, z_dim, redundancy_factor, matrix_path=None):
        super().__init__()
        self.z_dim = z_dim
        self.redundancy_factor = redundancy_factor
        self.x_dim = z_dim * redundancy_factor
        
        # 加载生成矩阵 G
        G = self._load_generation_matrix(matrix_path, z_dim, self.x_dim)
        
        # 关键点：将 G 注册为 buffer
        # 意味着它会随模型保存/加载，但不会被优化器更新 (requires_grad=False)
        self.register_buffer('G', G)

    def _load_generation_matrix(self, path, z_dim, x_dim):
        """
        从 .pkl 文件加载矩阵，或者在调试模式下生成单位矩阵
        """
        if path and os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    # codedVAE 的 pkl 格式通常是 numpy 数组
                    G_numpy = pickle.load(f)
                
                # 确保维度匹配 (原代码可能有转置问题，需检查 shape)
                G_tensor = torch.tensor(G_numpy, dtype=torch.float32)
                
                # 期望形状: [Z_dim, X_dim]
                if G_tensor.shape != (z_dim, x_dim):
                    # 如果形状反了，尝试转置
                    if G_tensor.T.shape == (z_dim, x_dim):
                        G_tensor = G_tensor.T
                    else:
                        raise ValueError(f"矩阵维度不匹配: 期望 ({z_dim}, {x_dim}), 实际 {G_tensor.shape}")
                
                print(f"✅ 成功加载 ECC 矩阵: {path}")
                return G_tensor
            
            except Exception as e:
                print(f"⚠️ 加载矩阵失败: {e}，将回退到随机矩阵。")
        
        else:
            print(f"ℹ️ 未找到矩阵文件或处于 Mock 模式，使用重复码逻辑生成 G。")
        
        # 如果没有文件，生成一个简单的重复码矩阵作为占位
        # 例如 L=2: [I, I]
        # 使用 Kronecker Product 生成符合论文截图的结构
        # G = I (x) [1, 1, ..., 1]
        # 结果形式:
        # [1, 1, 0, 0]
        # [0, 0, 1, 1]
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