import scipy.io
import os

path = "data/caltech/Caltech101-7.mat"
if os.path.exists(path):
    try:
        mat = scipy.io.loadmat(path)
        print("Keys:", mat.keys())
        for k in mat.keys():
            if not k.startswith('__'):
                val = mat[k]
                print(f"{k}: type={type(val)}, shape={val.shape if hasattr(val, 'shape') else 'scalar'}")
    except Exception as e:
        print(e)
else:
    print("File not found")
