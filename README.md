# AutoPartDetector v6
## AI-Based Part Identification System — Intel CPU Optimized

---

## System Requirements
- Python 3.9 – 3.11
- Intel i5 (any gen) / Intel UHD Graphics
- No NVIDIA GPU needed
- Windows 10/11
- RAM: 4 GB minimum, 8 GB recommended

---

## Installation

Run install_cpu.bat (double-click) or manually:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics==8.2.18 opencv-python Pillow numpy PyYAML
pip install openvino onnxruntime
```

Run the app:
```
python main.py
```

---

## Inference Speed on Intel i5-10310U

| Backend       | FPS (approx) | How it activates              |
|---------------|-------------|-------------------------------|
| OpenVINO .xml | 25–60 FPS   | Auto after training           |
| ONNX .onnx    | 12–25 FPS   | Auto if OpenVINO unavailable  |
| PyTorch .pt   | 5–12 FPS    | Fallback                      |

Training auto-exports to OpenVINO and ONNX — no manual steps.

---

## Full Workflow

### Step 1 — Add Part Info
Menu → Add Part → Part Info tab
- Fill: Part No, Part Name, Model, Supplier
- YOLO Class: short code (grab_handle, door_lock, etc.)
- Click Save Part Info

### Step 2 — Add Images
Menu → Add Part → Images & Annotations tab

Option A: Upload photos
- Click Upload Images → select 20–50 JPG/PNG files

Option B: Camera capture
- Click Camera Capture → open camera → click Capture from multiple angles

### Step 3 — Annotate (Two Methods)

Method A: Draw boxes manually
- Click Annotate on any image
- Left-drag to draw bounding box around the part
- Right-click a box to delete it
- Click Save Annotation

Method B: Upload existing YOLO .txt files
- Click Upload YOLO .txt button
- Choose Option A (auto-match by filename) or Option B (select pairs)
- Files are matched by base filename: handle_001.jpg ↔ handle_001.txt

YOLO .txt format (one line per box):
```
class_id  cx  cy  width  height
0 0.512345 0.498765 0.234567 0.456789
```
All values normalized 0.0–1.0. class_id is always 0 (remapped at training).

### Step 4 — Train
- Click Train AI button (in Images tab or top menu)
- Set Epochs: 50 (start), 100 (better accuracy)
- Set Image Size: 416 (faster) or 640 (more accurate)
- Click Start Training
- Wait ~10–30 minutes on Intel i5
- Model auto-exports to OpenVINO + ONNX

### Step 5 — Detect
- Menu → Home → Start Camera
- Show part to camera
- Bounding box appears with part name + part number
- Click the box → full details in right panel

---

## Accuracy Guide

| mAP50 | Meaning          | Action                          |
|-------|------------------|---------------------------------|
| 0.85+ | Excellent        | Ready for production            |
| 0.65+ | Good             | Add more images to improve      |
| 0.40+ | Moderate         | Need 30+ images per part        |
| <0.40 | Low              | Need more images + better boxes |

### Tips for Better Accuracy

1. Images
   - Minimum 20 images per part, ideally 50+
   - Capture from front, side, top, 45-degree angle
   - Vary distance: close-up, medium, far
   - Vary lighting: bright, dim, indoor, outdoor
   - Plain background preferred

2. Annotations
   - Draw box TIGHT around the part
   - Include the full part in the box
   - Avoid cutting off parts of the object
   - One box per part visible in frame

3. Training
   - Start with 50 epochs
   - If mAP50 < 0.65, add more images and retrain with 100 epochs
   - Image size 640 gives better accuracy than 416

---

## Project Structure

```
AutoPartDetector/
├── main.py                     Entry point
├── install_cpu.bat             One-click CPU install
├── requirements.txt
├── training_pipeline.py        YOLOv8n train + ONNX + OpenVINO export
│
├── database/
│   └── db_manager.py          SQLite: parts, images, history
│
├── detection/
│   └── detector.py            OpenVINO → ONNX → PyTorch inference
│
├── gui/
│   └── app.py                 Full Tkinter GUI
│
├── parts_images/               Uploaded/captured part photos
├── dataset/
│   ├── images/train|val/       Training images (auto-populated)
│   ├── labels/train|val/       Training labels (auto-populated)
│   └── labels_raw/             Raw annotation .txt files
├── trained_model/
│   ├── best.pt                 PyTorch model
│   ├── best.onnx               ONNX model (faster)
│   └── best_openvino/          OpenVINO model (fastest on Intel)
├── screenshots/
└── training_logs/
```

---

## Build EXE

```
pip install pyinstaller
pyinstaller AutoPartDetector.spec --clean
```

Output: dist/AutoPartDetector/AutoPartDetector.exe

