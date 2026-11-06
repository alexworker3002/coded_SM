import os
import scipy.io
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
import urllib.request
from src.utils.data_utils import MultiViewDataset, source_metadata

def load_100leaves_data(data_dir="./data/100leaves", reduce_dim=False):
    """
    Load 100Leaves dataset.
    Downloads .mat file if not present.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    file_path = os.path.join(data_dir, "100Leaves.mat")

    # Potential sources (GitHub blob URLs converted to raw format)
    urls = [
         "https://github.com/ChuanbinZhang/Multi-view-datasets/raw/master/100Leaves.mat",
         "https://github.com/Jeaninezpp/Multi-view-clustering/raw/master/100Leaves.mat",
         "https://github.com/ZhiqiangXu/MvC_Data/raw/master/100Leaves.mat"
    ]
    
    if os.path.exists(file_path):
         if os.path.getsize(file_path) < 10 * 1024:
            print(f"⚠️ Found corrupt/empty file at {file_path}. Deleting and re-downloading...")
            os.remove(file_path)

    if not os.path.exists(file_path):
        print(f"Downloading 100Leaves dataset to {file_path}...")
        success = False
        
        for url in urls:
            print(f"Trying source: {url} ...")
            try:
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(url, file_path)
                
                if os.path.getsize(file_path) < 10 * 1024:
                    raise RuntimeError("Downloaded file is too small.")
                    
                print("Download complete.")
                success = True
                break
            except Exception as e:
                print(f"❌ Failed: {e}")
                if os.path.exists(file_path):
                     os.remove(file_path)

        if not success:
            raise FileNotFoundError(
                f"All 100Leaves downloads failed. Place real data at {file_path}. "
                "Synthetic data is never generated as a fallback."
            )

    try:
        mat = scipy.io.loadmat(file_path)
    except Exception as e:
         raise RuntimeError(f"Failed to load .mat file: {e}")

    label_key = 'Y' if 'Y' in mat else ('y' if 'y' in mat else None)
    if 'X' not in mat or label_key is None:
         keys = [k for k in mat.keys() if not k.startswith('__')]
         raise ValueError(f"Invalid .mat format: keys 'X' and 'Y'/'y' expected. Found: {keys}")

    raw_X = mat['X']
    
    if raw_X.shape[0] == 3 and raw_X.shape[1] == 1:
        raw_X = raw_X.flatten()
    elif raw_X.shape[0] == 1 and raw_X.shape[1] == 3:
        raw_X = raw_X[0]
        
    raw_Y = mat[label_key]
    
    labels = torch.tensor(raw_Y.flatten(), dtype=torch.long)
    if labels.min() == 1:
        labels -= 1
        
    view_names = ["shape", "texture", "margin"]
    
    processed_views = {}
    scaler = StandardScaler()
    
    print(f"Preprocessing 100Leaves (N={len(labels)})...")
    
    for i, name in enumerate(view_names):
        if i >= len(raw_X):
            break
        
        data = raw_X[i].astype(np.float32)
        
        if reduce_dim and data.shape[1] > 100:
             pca = PCA(n_components=100, random_state=42)
             data = pca.fit_transform(data)
             
        data = scaler.fit_transform(data)
        processed_views[name] = torch.tensor(data, dtype=torch.float32)
        
    return MultiViewDataset(processed_views, labels, metadata=source_metadata([file_path]))
