@echo off
echo ============================================
echo  AutoPartDetector - Build EXE
echo ============================================

echo [1/3] Installing dependencies...
pip install -r requirements.txt
pip install pyinstaller

echo [2/3] Building executable...
pyinstaller AutoPartDetector.spec --clean

echo [3/3] Done!
echo.
echo Output: dist\AutoPartDetector\AutoPartDetector.exe
echo.
pause
