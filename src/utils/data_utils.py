import os
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

class MultiViewDataset(Dataset):
    def __init__(self, views_dict, labels):
        self.views = views_dict
        self.labels = labels
        self.view_names = list(views_dict.keys())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample_views = {name: self.views[name][idx] for name in self.view_names}
        # Refactored to return index for GPLVM
        return sample_views, self.labels[idx], idx

def load_mfeat_data(data_dir="./data/raw", mode="real"):
    """
    mode: "real" 从文件加载, "mock" 生成随机数据
    """
    # mfeat 数据集的标准特征维度 (根据 UCI 官方定义)
    view_dims = {
        "fac": 216, "fou": 76, "kar": 64, 
        "pix": 240, "zer": 47, "mor": 6
    }
    
    processed_views = {}
    num_samples = 2000 # mfeat 共有 2000 个样本 (10类 x 200)

    if mode == "mock":
        print("💡 [Mock Mode] 正在生成模拟多视图数据以供调试...")
        for view_name, dim in view_dims.items():
            # 生成均值为0，方差为1的随机数，模拟标准化后的数据
            processed_views[view_name] = torch.randn(num_samples, dim)
        labels = torch.repeat_interleave(torch.arange(10), 200)
        
    else:
        # 确保目录存在
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"找不到数据目录 {data_dir}，请先运行下载脚本或开启 mode='mock'")
            
        scaler = StandardScaler()
        for view_name, file_name in view_dims.items():
            path = os.path.join(data_dir, f"mfeat-{view_name}")
            # 真实读取逻辑
            raw_data = np.loadtxt(path)
            standardized_data = scaler.fit_transform(raw_data)
            processed_views[view_name] = torch.tensor(standardized_data, dtype=torch.float32)
        
        labels = torch.tensor(np.repeat(np.arange(10), 200), dtype=torch.long)
    
    return MultiViewDataset(processed_views, labels)


def get_dataset(dataset_name, data_dir=None, **kwargs):
    """
    Unified dataset factory function.
    Routes to the appropriate loader based on dataset_name.
    """
    dataset_name_lower = dataset_name.lower()
    
    if "mfeat" in dataset_name_lower:
        dir_path = data_dir if data_dir else "./data/raw"
        return load_mfeat_data(data_dir=dir_path, **kwargs)
    
    elif "caltech" in dataset_name_lower:
        from src.utils.data_caltech import load_caltech_data
        dir_path = data_dir if data_dir else "./data/caltech"
        return load_caltech_data(data_dir=dir_path, **kwargs)
    
    elif "100leaves" in dataset_name_lower:
        from src.utils.data_100leaves import load_100leaves_data
        dir_path = data_dir if data_dir else "./data/100leaves"
        return load_100leaves_data(data_dir=dir_path, **kwargs)
    
    elif "aloi" in dataset_name_lower:
        from src.utils.data_aloi import load_aloi_data
        dir_path = data_dir if data_dir else "./data/aloi"
        return load_aloi_data(data_dir=dir_path, **kwargs)
    
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")