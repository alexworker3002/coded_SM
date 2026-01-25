from src.utils.data_100leaves import load_100leaves_data
import torch

def verify():
    print("Verifying 100Leaves Data Loading...")
    try:
        ds = load_100leaves_data(data_dir="data/100leaves")
        print(f"✅ Successfully loaded dataset!")
        print(f"Samples: {len(ds)}")
        print(f"Views: {ds.view_names}")
        
        expected_views = ["shape", "texture", "margin"]
        for v in expected_views:
            if v not in ds.view_names:
                raise ValueError(f"Missing view: {v}")
            shape = ds.views[v].shape
            print(f"  - {v}: {shape}")
            if shape[1] != 64:
                 raise ValueError(f"Incorrect dim for {v}: got {shape[1]}, expected 64")

        print(f"Labels shape: {ds.labels.shape}")
        print(f"Label range: {ds.labels.min().item()} - {ds.labels.max().item()}")
        
        if ds.labels.min() < 0 or ds.labels.max() > 99:
             raise ValueError("Labels out of expected range 0-99")
             
        print("\nVerification Passed. Code is ready.")
    except Exception as e:
        print(f"❌ Verification Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
