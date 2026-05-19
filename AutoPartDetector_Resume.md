# AutoPartDetector — Project Analysis & Resume Description

---

## 📌 One-Line Summary (For Resume Header)
**AI-Powered Industrial Part Identification System** — Real-time object detection desktop application for automobile manufacturing quality control.

---

## 🏆 Resume Entry (Copy-Paste Ready)

### Project: AutoPartDetector — AI Industrial Part Identification System
**Tech Stack:** Python · YOLOv8 (Ultralytics) · OpenCV · PyQt5/Tkinter · SQLite · Intel OpenVINO · ONNX Runtime · PyInstaller

- Designed and built a **production-ready, standalone Windows desktop application** for real-time AI-based identification of automobile parts in an industrial inspection environment.
- Engineered a **custom YOLOv8 object detection pipeline** (confidence 0.50, NMS IOU 0.45) with **agnostic NMS**, bounding-box size filtering, and **exponential moving average (EMA) FPS smoothing** for stable, noise-free inference.
- Implemented an **offline-first architecture** — the entire system (model loading, training, inference) functions without internet; supports Intel CPU-only deployment via **OpenVINO** (25–60 FPS), **ONNX Runtime** (12–25 FPS), and PyTorch fallback.
- Built a **full in-app model training pipeline** with: per-class 80/20 dataset splitting, automatic offline data augmentation (brightness, contrast, blur, noise, hue shift, shadow, horizontal flip), AdamW optimizer, cosine LR schedule, mosaic/mixup/copy-paste augmentation, and post-training validation grading (A/B/C/D based on mAP50).
- Developed a **Visual Similarity Engine** using a 1184-dimensional feature vector (HSV colour histogram + Canny edge map + Sobel gradient magnitude blocks), cosine similarity, and disk-cached feature vectors — enabling visual part lookup without metadata.
- Designed and implemented a **3-table SQLite database** (parts, part_images, detection_history) with full CRUD operations, YOLO class mapping, annotation tracking, and detection history logging.
- Built a **multi-tab industrial GUI** (Tkinter) with: live camera feed with L-shaped corner bracket overlays, persistent seen-once detection list (2.5s debounce window), card-based part display, manual selection, reset functionality, and a scrollable detection history panel.
- Created a **tablet/touchscreen-optimized UX layer** (`tablet_utils.py`) including a floating virtual on-screen keyboard (singleton), global mouse-wheel scroll binding, and finger-swipe touch scrolling for canvas elements.
- Packaged the complete application as a **standalone Windows executable** using PyInstaller (`.spec` configuration), enabling zero-dependency deployment on factory floor systems.

---

## 🔍 Full Technical Analysis

### Project Overview
AutoPartDetector is a **CPU-optimized, offline-capable, industrial-grade AI inspection tool** for real-time identification and cataloguing of automobile parts. It combines computer vision, a custom-trained deep learning model, a relational parts database, and a professional touch-optimized GUI — all deployable as a single `.exe` on a standard Intel i5 machine.

---

### Architecture Overview

```
AutoPartDetector/
├── main.py                  → Entry point
├── gui/app.py               → Full multi-tab Tkinter GUI (~99KB, core UI)
├── detection/detector.py    → YOLOv8 inference engine
├── database/db_manager.py   → SQLite CRUD layer
├── training_pipeline.py     → Full model training orchestrator
├── similarity_engine.py     → Visual part-similarity search
├── tablet_utils.py          → Touch/scroll/keyboard utilities
├── requirements.txt         → CPU-only dependency manifest
└── AutoPartDetector.spec    → PyInstaller bundle config
```

---

### Module-by-Module Breakdown

#### 1. `detection/detector.py` — Inference Engine
| Feature | Detail |
|---|---|
| Model | YOLOv8 custom-trained `best.pt` (falls back to base model) |
| Confidence | 0.50 minimum threshold |
| NMS IOU | 0.45 with agnostic NMS |
| Max detections | 20 per frame |
| Box filter | Removes detections < 1% or > 92% of frame area (eliminates background noise) |
| FPS tracking | Exponential moving average (EMA, α=0.1) for smooth display |
| Overlay | L-shaped corner brackets on all 4 corners + label tag with part name, part no., confidence % |
| Compatibility fix | PyTorch 2.6+ `weights_only` patch via `functools.wraps` |
| Config isolation | Custom `.ultralytics_cfg` directory to avoid drive conflicts |

#### 2. `training_pipeline.py` — Model Training Orchestrator
| Feature | Detail |
|---|---|
| Base model | YOLOv8s (offline-first from `trained_model/pretrained/`) |
| Dataset split | Per-class balanced 80/20 (train/val) |
| Augmentation | Offline: brightness, contrast, blur, noise, hue shift, shadow, horizontal flip |
| Target coverage | 30 min images per class → auto-augments up to 60 |
| Optimizer | AdamW, lr0=0.001, cosine LR decay |
| Augmentation flags | mosaic=1.0, mixup=0.1, copy_paste=0.1, degrees=15, scale=0.6 |
| Callbacks | Real-time per-epoch progress: mAP50, Precision, Recall |
| Post-training | Auto-copies best.pt, runs validation, exports JSON metrics log |
| Grading system | A (mAP50 ≥ 0.85), B (≥ 0.70), C (≥ 0.50), D (< 0.50) |

#### 3. `database/db_manager.py` — Data Layer
| Table | Columns | Purpose |
|---|---|---|
| `parts` | part_no, part_name, model, supplier, group_name, date, zone, quantity, judgement, reason, image_path, yolo_class | Master parts catalogue |
| `part_images` | part_no, image_path, label_path, annotated, source | Per-part training images + annotation status |
| `detection_history` | part_no, part_name, confidence, timestamp, screenshot_path, judgement | Audit trail of all detections |

Key operations: `add_part`, `delete_part`, `search_parts` (LIKE query), `get_part_by_yolo_class`, `get_all_annotated_images` (JOIN query for training), `log_detection`, `get_history`.

#### 4. `similarity_engine.py` — Visual Search
| Feature | Detail |
|---|---|
| Feature vector | 1184-dimensional: HSV histogram (96-d) + Canny edge map 32×32 (1024-d) + Sobel gradient 8×8 blocks (64-d) |
| Normalization | L2-normalized float32 |
| Similarity metric | Cosine distance |
| Tiers | EXACT (≥ 0.88), SIMILAR (≥ 0.62), SLIGHTLY (≥ 0.38) |
| Caching | Disk-cached JSON (keyed by part_no + file hash), in-memory LRU |
| Live search | Supports feature extraction directly from live camera bounding-box crop |
| Aggregation | Averages features across up to 5 images per part |

#### 5. `tablet_utils.py` — Touch UX Layer
| Feature | Detail |
|---|---|
| Virtual keyboard | Floating singleton Tkinter window, auto-positioned, CAPS toggle, placeholder-aware |
| Global scroll | Root-window `MouseWheel` binding → scrolls topmost visible registered canvas |
| Touch scroll | Finger-swipe (B1-Motion) on canvas with 8px drag threshold, units-based yview scroll |
| Cross-platform | Windows (`<MouseWheel>`), Linux (`<Button-4>`, `<Button-5>`) |
| Offline model loader | Checks 4 candidate paths before attempting download |

#### 6. `gui/app.py` — Main Application UI (~99KB)
| Panel/Tab | Feature |
|---|---|
| Home / Detection | Live webcam feed thread, L-bracket overlays, 2.5s debounce seen-once list, card-based detected parts |
| Part Management | Add/edit/delete parts, search, image upload, camera capture |
| Annotation Tool | Draw bounding boxes (left-drag), delete (right-click), save YOLO `.txt` |
| Training | Epoch/image-size config, real-time progress bar, mAP/grade result display |
| History | Scrollable detection log with timestamps and confidence |
| Reset | One-click clear of seen-once detection memory |

---

### Technology Stack

| Category | Technology | Version/Detail |
|---|---|---|
| Language | Python | 3.9 – 3.11 |
| AI / Detection | YOLOv8 (Ultralytics) | 8.2.18 |
| Deep Learning | PyTorch | CPU-only build |
| Inference Runtime | Intel OpenVINO | ≥ 2024.1.0 (25–60 FPS) |
| Inference Runtime | ONNX Runtime | ≥ 1.18.0 (12–25 FPS) |
| Computer Vision | OpenCV | ≥ 4.9.0 |
| GUI Framework | Tkinter (with custom touch layer) | Built-in Python |
| Database | SQLite | Via Python `sqlite3` |
| Image Processing | Pillow | ≥ 10.3.0 |
| Packaging | PyInstaller | .spec config |
| Target Hardware | Intel i5 / Intel UHD (no GPU) | Windows 10/11 |

---

### Key Engineering Decisions

1. **CPU-only deployment** — Entire inference chain uses OpenVINO → ONNX → PyTorch fallback, enabling deployment on factory-floor PCs without NVIDIA GPUs.
2. **Offline-first** — All model weights, training pipeline, and similarity engine work without any internet connection after initial setup.
3. **EMA FPS** — Exponential moving average (α=0.1) prevents FPS counter jitter on variable-load CPU inference.
4. **2.5-second debounce** — Prevents transient detections from polluting the seen-once list, ensuring only stable, confirmed detections appear.
5. **Visual similarity without metadata** — The 1184-d feature vector is purely image-based; no part numbers or names are used, enabling discovery of visually similar parts even without text search.
6. **Singleton virtual keyboard** — Only one floating keyboard exists at a time; it re-attaches to whichever Entry gains focus, preventing window accumulation.
7. **PyTorch 2.6 compatibility patch** — A `functools.wraps` monkey-patch on `torch.load` defaults `weights_only=False`, maintaining compatibility across PyTorch versions.

---

## 📄 Short Resume Bullet Points (Choose 3–5)

> Use these as bullet points under a "Projects" section on your resume:

- Built a **real-time AI automobile part identification system** using YOLOv8 and OpenCV, achieving 25–60 FPS on Intel CPU hardware via OpenVINO optimization — deployable as a standalone Windows executable.
- Developed an **end-to-end ML pipeline** from image annotation (custom YOLO bounding-box editor) through offline data augmentation, model training (AdamW + cosine LR + mosaic augmentation), and validated performance grading (mAP50-based A/B/C/D system).
- Designed a **1184-dimensional visual feature similarity engine** (HSV histogram + Canny edges + Sobel gradients, cosine similarity, disk-cached) for metadata-free part lookup.
- Engineered a **production-ready, offline-first desktop GUI** (Tkinter) with live webcam detection overlays, SQLite parts database, detection history logging, and a full touch/tablet UX layer (virtual keyboard, swipe scroll, global mouse-wheel).
- Packaged a complex Python AI application as a **single-file Windows executable** using PyInstaller with full asset and model bundling.

---

## 🎓 Skills Demonstrated (For Skills Section)

`Python` · `Machine Learning` · `Computer Vision` · `YOLOv8` · `Deep Learning` · `Object Detection` · `OpenCV` · `Intel OpenVINO` · `ONNX` · `PyTorch` · `SQLite` · `Desktop GUI Development` · `Tkinter` · `Data Augmentation` · `Model Training & Evaluation` · `PyInstaller` · `Software Architecture` · `Embedded/Industrial AI`
