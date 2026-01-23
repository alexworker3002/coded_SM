import torch
import random
import numpy as np

def get_device(config_device="auto"):
    """
    智能设备选择逻辑
    """
    if config_device != "auto":
        return torch.device(config_device)
        
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        # 针对你的 Mac 环境
        return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed=42):
    """
    保证科研实验的可复现性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)