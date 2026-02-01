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
    elif mode == "rotation":
        # Pass validation here, specific logic is handled later in the function
        view_configs = {} # Placeholder, not used for rotation mode
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
    
    # 1. Define cache path
    cache_name = f"cache_{mode}_{num_samples}.pt"
    cache_path = os.path.join(data_dir, cache_name)
    
    # 2. Try loading from cache
    if os.path.exists(cache_path):
        print(f"[ALOI] Found cache at {cache_path}. Loading...")
        try:
            cache_data = torch.load(cache_path)
            # Verify cache contains all views
            if all(v in cache_data['views'] for v in (view_configs.keys() if mode in ["illumination", "color"] else tar_paths.keys())):
                print(f"[ALOI] Cache loaded successfully. {num_samples} samples.")
                return MultiViewDataset(cache_data['views'], cache_data['labels'])
        except Exception as e:
            print(f"⚠️  Cache load failed: {e}. Re-loading from scratch.")

    # 3. Initialize storage and progress tracking
    processed_views = {}
    from tqdm import tqdm
    
    # Check if raw data exists for faster loading
    raw_root = os.path.join(data_dir, "raw", "png4")
    use_raw = os.path.exists(raw_root)
    if use_raw:
        print(f"[ALOI] Found extracted data at {raw_root}. Using fast disk loading.")
    else:
        print(f"[ALOI] Extracted data not found at {raw_root}. Falling back to slow TAR loading.")

    if mode in ["illumination", "color"]:
        for view_name, suffix in view_configs.items():
            images = []
            print(f"  Extracting view: {view_name}...")
            if use_raw:
                # Fast path: Disk
                for obj_id in tqdm(object_ids, desc=f"Loading {view_name}"):
                    img_path = os.path.join(raw_root, str(obj_id), f"{obj_id}{suffix}.png")
                    try:
                        img = Image.open(img_path).convert('RGB')
                        images.append(transform(img))
                    except FileNotFoundError:
                        images.append(torch.zeros(3, 144, 192))
            else:
                # Slow path: TAR
                with tarfile.open(tar_path, 'r') as tar:
                    for obj_id in tqdm(object_ids, desc=f"Loading {view_name}"):
                        filename = f"png4/{obj_id}/{obj_id}{suffix}.png"
                        try:
                            member = tar.getmember(filename)
                            f = tar.extractfile(member)
                            img = Image.open(f).convert('RGB')
                            images.append(transform(img))
                        except KeyError:
                            images.append(torch.zeros(3, 144, 192))
            
            processed_views[view_name] = torch.stack(images)  # (N, 3, 144, 192)
            print(f"  ✓ {view_name}: {processed_views[view_name].shape}")
    
    elif mode == "mixed":
        for view_name, (tar_path, suffix) in tar_paths.items():
            images = []
            print(f"  Extracting view: {view_name}...")
            if use_raw:
                for obj_id in tqdm(object_ids, desc=f"Loading {view_name}"):
                    img_path = os.path.join(raw_root, str(obj_id), f"{obj_id}{suffix}.png")
                    try:
                        img = Image.open(img_path).convert('RGB')
                        images.append(transform(img))
                    except FileNotFoundError:
                        images.append(torch.zeros(3, 144, 192))
            else:
                with tarfile.open(tar_path, 'r') as tar:
                    for obj_id in tqdm(object_ids, desc=f"Loading {view_name}"):
                        filename = f"png4/{obj_id}/{obj_id}{suffix}.png"
                        try:
                            member = tar.getmember(filename)
                            f = tar.extractfile(member)
                            img = Image.open(f).convert('RGB')
                            images.append(transform(img))
                        except KeyError:
                            images.append(torch.zeros(3, 144, 192))
            
            processed_views[view_name] = torch.stack(images)
            print(f"  ✓ {view_name}: {processed_views[view_name].shape}")

    elif mode == "rotation":
        # Harder task: Different rotations are treated as separate samples (N becomes 4 * num_objects)
        # Sample 1: Front (0) -> Views: -5, 0, +5
        # Sample 2: Right (90) -> Views: 85, 90, 95
        # Sample 3: Back (180) -> Views: 175, 180, 185
        # Sample 4: Left (270) -> Views: 265, 270, 275
        
        sectors = {
            "front": [355, 0, 5],
            "right": [85, 90, 95],
            "back": [175, 180, 185],
            "left": [265, 270, 275]
        }
        
        # View names are "angle_minus5", "angle_0", "angle_plus5"
        view_names = ["view_left", "view_center", "view_right"]
        
        # Initialize lists for all views
        all_view_data = {v: [] for v in view_names}
        expanded_labels = []
        
        print(f"[ALOI] Rotation Mode: Generating 4 samples per object (Front/Right/Back/Left)...")
        
        if use_raw:
            # Fast path: Disk
            for obj_id in tqdm(object_ids, desc="Loading Rotations"):
                # For each object, generate 4 samples
                for sector_name, angles in sectors.items():
                    # Add label (same label for all 4 sectors of this object)
                    expanded_labels.append(obj_id - 1)
                    
                    # Load 3 views for this sector
                    for i, angle in enumerate(angles):
                        angle_norm = angle % 360  # Handle 360 -> 0 if needed (though files use r0, r5...)
                        img_path = os.path.join(raw_root, str(obj_id), f"{obj_id}_r{angle_norm}.png")
                        try:
                            img = Image.open(img_path).convert('RGB')
                            all_view_data[view_names[i]].append(transform(img))
                        except FileNotFoundError:
                            # Try alternate naming if needed, or zero pad
                            print(f"Warning: Missing {img_path}")
                            all_view_data[view_names[i]].append(torch.zeros(3, 144, 192))
        else:
             print("Error: Rotation mode currently requires extracted raw data for speed.")
             return None

        # Stack tensors
        for v in view_names:
            processed_views[v] = torch.stack(all_view_data[v])
            print(f"  ✓ {v}: {processed_views[v].shape}")
            
        # Update labels to be the expanded list
        # We assign to local variable 'labels' which will be returned
        # Note: We need to override the default labels created at the end of function
        final_labels = torch.tensor(expanded_labels, dtype=torch.long)
        
        print(f"[ALOI] Saving processed data to cache: {cache_path}...")
        torch.save({'views': processed_views, 'labels': final_labels}, cache_path)
        
        print(f"[ALOI] Loading complete: {len(final_labels)} samples (4x objects), {len(processed_views)} views.")
        return MultiViewDataset(processed_views, final_labels)

    # 4. Finalize and Cache
    labels = torch.tensor(object_ids, dtype=torch.long) - 1  # 0-indexed labels
    
    print(f"[ALOI] Saving processed data to cache: {cache_path}...")
    torch.save({'views': processed_views, 'labels': labels}, cache_path)
    
    print(f"[ALOI] Loading complete: {num_samples} objects, {len(processed_views)} views.")
    return MultiViewDataset(processed_views, labels)
