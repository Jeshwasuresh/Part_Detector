"""
Detection Engine — AutoPartDetector
====================================
• Loads custom trained best.pt if present
• Falls back to base model from offline cache
• NO internet needed after first setup
• Confidence 0.50, NMS IOU 0.45, agnostic NMS
• Box size filter removes background false positives
• Smooth EMA FPS
"""
import cv2, os, sys, time
import numpy as np

# ── Fix 1: redirect Ultralytics config away from any stale/missing drive ─────
_CFG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        '.ultralytics_cfg')
os.makedirs(_CFG_DIR, exist_ok=True)
os.environ['YOLO_CONFIG_DIR'] = _CFG_DIR   # force-override any system-level value

# ── Fix 2: PyTorch 2.6+ weights_only=True breaks Ultralytics .pt loading ─────
try:
    import torch, functools
    _orig_torch_load = torch.load
    @functools.wraps(_orig_torch_load)
    def _patched_torch_load(f, *args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return _orig_torch_load(f, *args, **kwargs)
    torch.load = _patched_torch_load
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────

try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEST_MODEL = os.path.join(ROOT, 'trained_model', 'best.pt')

BBOX_COLORS = [
    (0, 220, 0),    (255, 100, 0),  (0, 80, 255),
    (180, 0, 200),  (0, 165, 255),  (0, 210, 210),
    (200, 200, 0),  (255, 0, 180),
]

CONF   = 0.50    # minimum confidence
IOU    = 0.45    # NMS IOU
MIN_FR = 0.01    # minimum box area as fraction of frame
MAX_FR = 0.92    # maximum box area as fraction of frame


class PartDetector:
    def __init__(self):
        self.model       = None
        self.model_ready = False
        self.model_info  = "Not loaded"
        self.classes     = []
        self._fps_ema    = 0.0
        self.last_fps    = 0
        self._load()

    def reload(self):
        self.model       = None
        self.model_ready = False
        self.model_info  = "Reloading..."
        self._load()

    def _load(self):
        if not YOLO_OK:
            self.model_info = "ultralytics not installed"
            return

        # 1. Custom trained model
        if os.path.exists(BEST_MODEL):
            try:
                self.model       = YOLO(BEST_MODEL)
                self.model_ready = True
                self.classes     = list(self.model.names.values())
                self.model_info  = f"Custom model  {len(self.classes)} class(es): {self.classes}"
                print(f"[Detector] {self.model_info}")
                return
            except Exception as e:
                print(f"[Detector] Custom model error: {e}")

        # 2. No custom model
        self.model_ready = False
        self.model_info  = "⚠ No trained model — annotate images and Train first"
        print("[Detector]", self.model_info)

    def is_ready(self):
        return self.model_ready

    # ── Inference ─────────────────────────────────────────────────────────────
    def detect(self, frame):
        t0 = time.perf_counter()
        results = []

        if self.model and self.model_ready:
            h, w   = frame.shape[:2]
            f_area = h * w
            try:
                res = self.model.predict(
                    frame,
                    verbose      = False,
                    conf         = CONF,
                    iou          = IOU,
                    agnostic_nms = True,
                    max_det      = 20,
                )[0]
                for box in res.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf   = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label  = res.names.get(cls_id, str(cls_id))
                    area   = max(0, x2-x1) * max(0, y2-y1)
                    frac   = area / f_area
                    if frac < MIN_FR or frac > MAX_FR:
                        continue
                    results.append({
                        'label': label, 'confidence': conf,
                        'bbox': (x1, y1, x2, y2),
                    })
            except Exception as e:
                print(f"[Detector] predict: {e}")

        dt = time.perf_counter() - t0
        if dt > 0:
            inst = 1.0 / dt
            self._fps_ema = (0.1*inst + 0.9*self._fps_ema
                             if self._fps_ema > 0 else inst)
            self.last_fps = round(self._fps_ema, 1)

        return results

    # ── Draw ──────────────────────────────────────────────────────────────────
    def draw_overlay(self, frame, detections, lookup_fn):
        overlay = []
        h_f, w_f = frame.shape[:2]

        for idx, det in enumerate(detections):
            label            = det['label']
            conf             = det['confidence']
            x1, y1, x2, y2  = det['bbox']
            part             = lookup_fn(label)
            color            = BBOX_COLORS[idx % len(BBOX_COLORS)]

            pno   = part['part_no']  if part else label.upper()
            pname = (part['part_name'].split('/')[0].strip()[:20]
                     if part else label.replace('_', ' ').title())

            # ── Corner bracket annotations on all four corners ─────────────
            bw      = x2 - x1
            bh      = y2 - y1
            arm     = max(12, min(30, int(min(bw, bh) * 0.18)))  # arm length
            thick_c = 3   # corner line thickness
            # semi-transparent full box (dim guide)
            overlay_img = frame.copy()
            cv2.rectangle(overlay_img, (x1, y1), (x2, y2), color, 1)
            cv2.addWeighted(overlay_img, 0.25, frame, 0.75, 0, frame)

            # Top-left corner
            cv2.line(frame, (x1, y1), (x1 + arm, y1), color, thick_c)
            cv2.line(frame, (x1, y1), (x1, y1 + arm), color, thick_c)
            # Top-right corner
            cv2.line(frame, (x2, y1), (x2 - arm, y1), color, thick_c)
            cv2.line(frame, (x2, y1), (x2, y1 + arm), color, thick_c)
            # Bottom-left corner
            cv2.line(frame, (x1, y2), (x1 + arm, y2), color, thick_c)
            cv2.line(frame, (x1, y2), (x1, y2 - arm), color, thick_c)
            # Bottom-right corner
            cv2.line(frame, (x2, y2), (x2 - arm, y2), color, thick_c)
            cv2.line(frame, (x2, y2), (x2, y2 - arm), color, thick_c)

            # ── Label tag ─────────────────────────────────────────────────
            font, fscl, thick, pad = cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1, 4
            L1 = pname
            L2 = f"{pno}  {int(conf*100)}%"
            (w1,h1),_ = cv2.getTextSize(L1, font, fscl, thick)
            (w2,h2),_ = cv2.getTextSize(L2, font, fscl, thick)
            tw = max(w1, w2) + pad*2
            th = h1 + h2 + pad*3
            ty = y1 - th - 4
            if ty < 0: ty = y2 + 4
            tx = max(0, min(x1, w_f - tw))
            # Rounded-looking label background
            cv2.rectangle(frame, (tx, ty), (tx+tw, ty+th), color, -1)
            cv2.rectangle(frame, (tx, ty), (tx+tw, ty+th), (255,255,255), 1)
            cv2.putText(frame, L1, (tx+pad, ty+h1+pad),
                        font, fscl, (0, 0, 0), thick+1, cv2.LINE_AA)
            cv2.putText(frame, L1, (tx+pad, ty+h1+pad),
                        font, fscl, (255,255,255), thick, cv2.LINE_AA)
            cv2.putText(frame, L2, (tx+pad, ty+h1+h2+pad*2),
                        font, fscl, (0, 0, 0), thick+1, cv2.LINE_AA)
            cv2.putText(frame, L2, (tx+pad, ty+h1+h2+pad*2),
                        font, fscl, (255,255,255), thick, cv2.LINE_AA)

            overlay.append({'part': part, 'label': label,
                            'bbox': (x1,y1,x2,y2), 'conf': conf})

        return frame, overlay
