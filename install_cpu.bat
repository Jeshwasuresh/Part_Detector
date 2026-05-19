@echo off
echo ============================================================
echo  AutoPartDetector - CPU Install (Intel i5 / Intel UHD)
echo ============================================================

echo [1/4] Installing CPU-only PyTorch...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo [2/4] Installing core packages...
pip install ultralytics==8.2.18 opencv-python Pillow numpy PyYAML

echo [3/4] Installing Intel OpenVINO (fastest on Intel CPU/iGPU)...
pip install openvino onnxruntime

echo [4/4] Done!
echo.
echo Run with:  python main.py
echo.
pause
