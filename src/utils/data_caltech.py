import os
import scipy.io
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
import urllib.request
from src.utils.data_utils import MultiViewDataset

def load_caltech_data(data_dir="./data/caltech", reduce_dim=True):
    """
    Load Caltech101-7 dataset.
    Downloads .mat file if not present.
    Performs PCA on high-dimensional views if reduce_dim is True.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    file_path = os.path.join(data_dir, "Caltech101-7.mat")
    
    # Mirror URL for Caltech101-7 .mat file
    # Source: https://github.com/yeqinglee/mvdata
    url = "https://github.com/yeqinglee/mvdata/raw/master/Caltech101-7.mat"
    
    if not os.path.exists(file_path):
        print(f"Downloading Caltech101-7 dataset to {file_path}...")
        try:
            urllib.request.urlretrieve(url, file_path)
            print("Download complete.")
        except Exception as e:
            raise RuntimeError(f"Download failed. Please manually download Caltech101-7.mat to {data_dir}. Error: {e}")

    # Load .mat file
    try:
        mat = scipy.io.loadmat(file_path)
    except Exception as e:
         raise RuntimeError(f"Failed to load .mat file: {e}")
    
    # Structure of Caltech101-7.mat from yeqinglee/mvdata:
    # X: (1, 6) object array. Each element is (1474, dim)
    # Y: (1474, 1) labels
    
    if 'X' not in mat or 'Y' not in mat:
         # Fallback check for other versions
         raise ValueError("Invalid .mat format: keys 'X' and 'Y' expected.")

    raw_X = mat['X'][0]  # Object array containing 6 views
    raw_Y = mat['Y']     # Labels
    
    # Convert labels
    labels = torch.tensor(raw_Y.flatten(), dtype=torch.long)
    # Ensure 0-based labels
    if labels.min() == 1:
        labels -= 1

    # View names corresponding to the order in common Caltech101-7 versions
    # Order: Gabor(48), WM(40), CENTRIST(254), HOG(1984), GIST(512), LBP(928)
    view_names = ["gabor", "wm", "centrist", "hog", "gist", "lbp"]
    
    processed_views = {}
    scaler = StandardScaler()

    print(f"Preprocessing Caltech101-7 (N={len(labels)})...")
    
    for i, name in enumerate(view_names):
        if i >= len(raw_X):
            break
            
        data = raw_X[i].astype(np.float32)
        original_dim = data.shape[1]
        
        # 1. PCA Dimensionality Reduction
        # Target dim 100 for high-dim views to enable efficient RFF approximation
        if reduce_dim and original_dim > 100:
            target_dim = 100
            print(f"  - View '{name}': PCA {original_dim} -> {target_dim}")
            pca = PCA(n_components=target_dim, random_state=42)
            data = pca.fit_transform(data)
        else:
             print(f"  - View '{name}': Keep dim {original_dim}")
        
        # 2. Standardization
        data = scaler.fit_transform(data)
        processed_views[name] = torch.tensor(data, dtype=torch.float32)

    return MultiViewDataset(processed_views, labels)
