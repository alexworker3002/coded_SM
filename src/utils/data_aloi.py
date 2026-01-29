import os
import tarfile
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from src.utils.data_utils import MultiViewDataset

def load_aloi_data(data_dir="./data/aloi", mode="illumination", num_objects=1000, reduce_objects=None):
    """
    Load ALOI (Amsterdam Library of Object Images) dataset.
    
    Args:
        data_dir: Directory containing ALOI tar files
        mode: Which collection to use as multi-view setup:
              - "illumination": Use different illumination directions (24 images per object)
              - "color": Use different illumination colors (12 images per object)
              - "mixed": Combine illumination + color for richer views
        num_objects: Number of objects to load (max 1000)
        reduce_objects: If set, randomly sample this many objects for faster experiments
    
    Returns:
        MultiViewDataset with image tensors of shape (N, 3, 144, 192)
    """
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"ALOI data directory not found: {data_dir}")
    
    # Define view configurations based on mode
    if mode == "illumination":
        # Use 4 different light directions as 4 views
        tar_path = os.path.join(data_dir, "aloi_red4_ill.tar")
        view_configs = {
            "light_dir1": "_l1c2",  # Light position 1, color 2
            "light_dir2": "_l3c2",  # Light position 3, color 2
            "light_dir3": "_l5c2",  # Light position 5, color 2
            "light_dir4": "_l7c2"   # Light position 7, color 2
        }
    elif mode == "color":
        # Use 4 different illumination colors as 4 views
        tar_path = os.path.join(data_dir, "aloi_red4_col.tar")
        view_configs = {
            "color1": "_i110",
            "color2": "_i140",
            "color3": "_i170",
            "color4": "_i210"
        }
    elif mode == "mixed":
        # Combine both for 6 views (richer representation)
        tar_paths = {
            "light1": (os.path.join(data_dir, "aloi_red4_ill.tar"), "_l1c2"),
            "light2": (os.path.join(data_dir, "aloi_red4_ill.tar"), "_l4c2"),
            "light3": (os.path.join(data_dir, "aloi_red4_ill.tar"), "_l7c2"),
            "color1": (os.path.join(data_dir, "aloi_red4_col.tar"), "_i130"),
            "color2": (os.path.join(data_dir, "aloi_red4_col.tar"), "_i170"),
            "color3": (os.path.join(data_dir, "aloi_red4_col.tar"), "_i210")
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to (C, H, W) and scales to [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    # Determine object IDs to load
    if reduce_objects:
        object_ids = np.random.choice(range(1, num_objects + 1), size=reduce_objects, replace=False)
        object_ids = sorted(object_ids)
    else:
        object_ids = list(range(1, num_objects + 1))
    
    num_samples = len(object_ids)
    print(f"[ALOI] Loading {num_samples} objects in '{mode}' mode...")
    
    # Initialize storage
    processed_views = {}
    
    if mode in ["illumination", "color"]:
        # Single TAR file case
        print(f"[ALOI] Extracting from {tar_path}...")
        
        for view_name, suffix in view_configs.items():
            images = []
            
            with tarfile.open(tar_path, 'r') as tar:
                for obj_id in object_ids:
                    # Construct filename: png4/{obj_id}/{obj_id}{suffix}.png
                    filename = f"png4/{obj_id}/{obj_id}{suffix}.png"
                    
                    try:
                        member = tar.getmember(filename)
                        f = tar.extractfile(member)
                        img = Image.open(f).convert('RGB')
                        img_tensor = transform(img)  # (3, 144, 192)
                        images.append(img_tensor)
                    except KeyError:
                        print(f"⚠️  Missing file: {filename}, using zero placeholder")
                        images.append(torch.zeros(3, 144, 192))
            
            processed_views[view_name] = torch.stack(images)  # (N, 3, 144, 192)
            print(f"  ✓ {view_name}: {processed_views[view_name].shape}")
    
    elif mode == "mixed":
        # Multiple TAR files case
        for view_name, (tar_path, suffix) in tar_paths.items():
            images = []
            
            with tarfile.open(tar_path, 'r') as tar:
                for obj_id in object_ids:
                    filename = f"png4/{obj_id}/{obj_id}{suffix}.png"
                    
                    try:
                        member = tar.getmember(filename)
                        f = tar.extractfile(member)
                        img = Image.open(f).convert('RGB')
                        img_tensor = transform(img)
                        images.append(img_tensor)
                    except KeyError:
                        print(f"⚠️  Missing file: {filename}")
                        images.append(torch.zeros(3, 144, 192))
            
            processed_views[view_name] = torch.stack(images)
            print(f"  ✓ {view_name}: {processed_views[view_name].shape}")
    
    # Labels: Each object is a class
    labels = torch.tensor(object_ids, dtype=torch.long) - 1  # Convert to 0-indexed
    
    print(f"[ALOI] Loaded {num_samples} objects with {len(processed_views)} views")
    return MultiViewDataset(processed_views, labels)
