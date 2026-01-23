import os
import urllib.request

def download_mfeat(target_dir="./data/raw"):
    """
    自动从 UCI 仓库下载 mfeat 数据集的所有视图文件
    """
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mfeat/"
    files = [
        "mfeat-fac", "mfeat-fou", "mfeat-kar", 
        "mfeat-pix", "mfeat-zer", "mfeat-mor"
    ]
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"创建目录: {target_dir}")

    for f in files:
        target_path = os.path.join(target_dir, f)
        if not os.path.exists(target_path):
            print(f"正在下载 {f}...")
            urllib.request.urlretrieve(base_url + f, target_path)
        else:
            print(f"{f} 已存在，跳过。")
    print("所有视图下载完成。")

if __name__ == "__main__":
    download_mfeat()