# AutoPartDetector.spec
# Build command: pyinstaller AutoPartDetector.spec

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

# Collect ultralytics assets
datas = []
try:
    datas += collect_data_files('ultralytics')
except Exception:
    pass

a = Analysis(
    ['main.py'],
    pathex=[os.path.abspath('.')],
    binaries=[] ,
    datas=datas + [
        ('database', 'database'),
        ('detection', 'detection'),
        ('gui', 'gui'),
        ('screenshots', 'screenshots'),
        ('history', 'history'),
        ('trained_model', 'trained_model'),
        ('assets', 'assets'),
    ],
    hiddenimports=[
        'cv2',
        'PIL',
        'PIL._tkinter_finder',
        'ultralytics',
        'sqlite3',
        'tkinter',
        'tkinter.ttk',
        'tkinter.messagebox',
        'tkinter.filedialog',
        'numpy',
        'database.db_manager',
        'detection.detector',
        'gui.app',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AutoPartDetector',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,   # Set True for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico',  # Add your icon file here
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AutoPartDetector',
)
