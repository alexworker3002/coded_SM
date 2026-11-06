import torch
import random
import numpy as np
import hashlib
import os
from pathlib import Path

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

def set_seed(seed=42, deterministic=True):
    """
    保证科研实验的可复现性
    """
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def dataset_fingerprint(dataset):
    """Hash the processed values, labels and their order, independent of file paths."""
    digest = hashlib.sha256()
    tensors = [('labels', dataset.labels)] + sorted(dataset.views.items())
    for name, tensor in tensors:
        digest.update(f'{name}:{tensor.dtype}:{tuple(tensor.shape)}'.encode())
        for chunk in tensor.detach().cpu().split(128):
            digest.update(chunk.contiguous().numpy().tobytes())
    return digest.hexdigest()


def source_fingerprint():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / 'src').rglob('*.py'))
    paths += [root / name for name in ('auto_analyze.py', 'create_experiment.py',
              'server_runner.py', 'run_reproducibility.py', 'requirements.txt', 'requirements-dev.txt')
              if (root / name).exists()]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
