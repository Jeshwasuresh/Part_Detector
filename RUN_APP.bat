@echo off
title AutoPartDetector - AI Part Recognition System
color 0A

echo.
echo  =====================================================
echo    AutoPartDetector - AI Part Recognition System
echo  =====================================================
echo.

:: ── Step 1: Check Python ──────────────────────────────
echo  [1/4] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    color 0C
    echo.
    echo  ERROR: Python is not installed or not in PATH.
    echo.
    echo  Please install Python 3.9+ from:
    echo    https://www.python.org/downloads/
    echo.
    echo  Make sure to check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo         Found: %%v
echo.

:: ── Step 2: Check if packages are already installed ──
echo  [2/4] Checking required packages...
python -c "import PyQt5" >nul 2>&1
set PYQT5_MISSING=%errorlevel%

python -c "import cv2" >nul 2>&1
set CV2_MISSING=%errorlevel%

python -c "import ultralytics" >nul 2>&1
set ULTRALYTICS_MISSING=%errorlevel%

if %PYQT5_MISSING%==0 if %CV2_MISSING%==0 if %ULTRALYTICS_MISSING%==0 (
    echo         All packages already installed. Skipping install.
    goto :launch
)

:: ── Step 3: Install missing packages ─────────────────
echo.
echo  [3/4] Installing / updating required packages...
echo         (This only happens once — please wait...)
echo.

python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo         Installing CPU-only PyTorch...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
    if errorlevel 1 (
        echo.
        echo  WARNING: PyTorch install had issues. Retrying without quiet mode...
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    )
)

if %PYQT5_MISSING% NEQ 0 (
    echo         Installing PyQt5...
    pip install PyQt5 --quiet
)

if %CV2_MISSING% NEQ 0 (
    echo         Installing OpenCV...
    pip install opencv-python --quiet
)

if %ULTRALYTICS_MISSING% NEQ 0 (
    echo         Installing Ultralytics (YOLOv8)...
    pip install ultralytics==8.2.18 --quiet
)

echo         Installing remaining dependencies...
pip install Pillow numpy PyYAML onnxruntime openvino --quiet

echo.
echo         Packages installed successfully!
echo.

:: ── Step 4: Launch App ───────────────────────────────
:launch
echo  [4/4] Launching AutoPartDetector...
echo.
echo  =====================================================
echo.

cd /d "%~dp0"
python main.py

:: ── Handle crash ─────────────────────────────────────
if errorlevel 1 (
    color 0C
    echo.
    echo  =====================================================
    echo   The application exited with an error.
    echo   See the error message above for details.
    echo  =====================================================
    echo.
    pause
) else (
    echo.
    echo  Application closed successfully.
    timeout /t 2 >nul
)
