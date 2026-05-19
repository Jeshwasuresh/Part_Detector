"""
offline_setup.py
================
Run this ONCE while connected to the internet.
Downloads everything needed to run fully offline:
  - YOLOv8s base model (for training)
  - YOLOv8n base model (alternative)
  - pip packages into local ./packages/ folder

After running this, the system works with NO internet.
"""
import os, sys, subprocess, urllib.request, shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, 'trained_model')
PKG_DIR   = os.path.join(ROOT, 'packages')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PKG_DIR,   exist_ok=True)

MODELS = {
    'yolov8n.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt',
    'yolov8s.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt',
    'yolov8m.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt',
}

PACKAGES = [
    'ultralytics', 'opencv-python', 'Pillow', 'numpy', 'PyYAML',
    'torch', 'torchvision', 'torchaudio',
]

def download_models():
    print("\n=== Downloading YOLO base models ===")
    for fname, url in MODELS.items():
        dst = os.path.join(MODEL_DIR, fname)
        if os.path.exists(dst):
            print(f"  ✅ {fname} already exists")
            continue
        print(f"  Downloading {fname}...", end='', flush=True)
        try:
            urllib.request.urlretrieve(url, dst)
            print(f" done ({os.path.getsize(dst)//1024//1024} MB)")
        except Exception as e:
            print(f" FAILED: {e}")

def download_packages():
    print("\n=== Downloading pip packages (offline cache) ===")
    cmd = [sys.executable, '-m', 'pip', 'download',
           '--dest', PKG_DIR,
           '--extra-index-url', 'https://download.pytorch.org/whl/cpu',
    ] + PACKAGES
    subprocess.run(cmd, check=False)
    print(f"\n  Packages cached to: {PKG_DIR}")

def install_offline():
    print("\n=== Installing from offline cache ===")
    cmd = [sys.executable, '-m', 'pip', 'install',
           '--no-index', '--find-links', PKG_DIR,
    ] + PACKAGES
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("  ✅ All packages installed offline")
    else:
        print("  Trying online install...")
        subprocess.run([sys.executable, '-m', 'pip', 'install'] + PACKAGES)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--install-only', action='store_true',
                   help='Install from existing cache (no download)')
    args = p.parse_args()

    if args.install_only:
        install_offline()
    else:
        download_models()
        download_packages()
        print("\n✅ Offline setup complete!")
        print("You can now disconnect from internet and run: python main.py")
