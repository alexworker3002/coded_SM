from src.utils.data_caltech import load_caltech_data
import torch

def verify():
    print("Verifying Caltech101-7 Data Loading...")
    try:
        ds = load_caltech_data(data_dir="data/caltech")
        print(f"✅ Successfully loaded dataset!")
        print(f"Samples: {len(ds)}")
        print(f"Views: {ds.view_names}")
        for k, v in ds.views.items():
            print(f"  - {k}: {v.shape}")
            
        # Check standard properties
        assert len(ds) == 1474, f"Expected 1474 samples, got {len(ds)}"
        assert 'gabor' in ds.views
        assert ds.views['gabor'].shape[1] == 48
        
        print("\nVerification Passed. Code is ready to push.")
    except Exception as e:
        print(f"❌ Verification Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
