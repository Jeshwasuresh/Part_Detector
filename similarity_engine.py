"""
Visual Similarity Engine
========================
Based ONLY on visual appearance of part images.
No metadata used — purely what the part LOOKS like.

Method:
  1. Colour histogram  (HSV, 3 channels × 32 bins  = 96-d)
  2. Edge/shape map    (Canny edges resized to 32×32 = 1024-d)
  3. Texture gradient  (Sobel magnitude blocks 8×8  = 64-d)
  Combined → 1184-d L2-normalised feature vector
  Similarity = cosine distance between feature vectors

Tiers:
  EXACT         score >= 0.88   (same part, same appearance)
  SIMILAR       score >= 0.62   (different variant, same family)
  SLIGHTLY      score >= 0.38   (loosely related shape/colour)

Features cached to disk (database/visual_cache.json).
Call invalidate(part_no) whenever images change.
"""

import cv2, os, sys, json, hashlib, shutil
import numpy as np
from pathlib import Path

ROOT       = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(ROOT, 'database', 'visual_cache.json')

TIER_EXACT   = 0.88
TIER_SIMILAR = 0.62
TIER_SLIGHT  = 0.38

# ── Feature dimensions ────────────────────────────────────────────────────────
_HIST_BINS  = 32   # per channel  →  3 × 32 = 96
_EDGE_SIZE  = 32   # edge map resized to 32×32 = 1024
_GRAD_CELLS = 8    # gradient block grid →  8×8 = 64
_FEAT_DIM   = 3 * _HIST_BINS + _EDGE_SIZE * _EDGE_SIZE + _GRAD_CELLS * _GRAD_CELLS


# ── Cache ─────────────────────────────────────────────────────────────────────
_mem_cache: dict = {}     # part_no  →  {'key': str, 'feat': list[float]}
_cache_dirty = False


def _load_cache():
    global _mem_cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                _mem_cache = json.load(f)
        except Exception:
            _mem_cache = {}


def _save_cache():
    global _cache_dirty
    if not _cache_dirty:
        return
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump(_mem_cache, f)
    _cache_dirty = False


def invalidate(part_no: str = None):
    """Remove cached features. Call after adding/deleting images."""
    global _mem_cache, _cache_dirty
    if part_no is None:
        _mem_cache = {}
    else:
        _mem_cache = {k: v for k, v in _mem_cache.items()
                      if not k.startswith(part_no + '::') and k != part_no}
    _cache_dirty = True
    _save_cache()


# ── Feature extraction ────────────────────────────────────────────────────────

def _extract(img_bgr: np.ndarray) -> np.ndarray:
    """
    Extract a 1184-d visual feature vector from a BGR image.
    Returns L2-normalised float32 array.
    """
    # 1. Resize to fixed canvas
    img = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)

    # 2. HSV colour histogram  (96-d)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_parts = []
    for ch in range(3):
        h = cv2.calcHist([hsv], [ch], None, [_HIST_BINS], [0, 256])
        h = cv2.normalize(h, h, norm_type=cv2.NORM_L1).flatten()
        hist_parts.append(h)
    hist_feat = np.concatenate(hist_parts)           # 96-d

    # 3. Canny edge map → resized to 32×32 (1024-d)
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred  = cv2.GaussianBlur(gray, (5, 5), 1.2)
    edges    = cv2.Canny(blurred, 50, 150).astype(np.float32) / 255.0
    edge_map = cv2.resize(edges, (_EDGE_SIZE, _EDGE_SIZE),
                          interpolation=cv2.INTER_AREA).flatten()    # 1024-d

    # 4. Sobel gradient magnitude → 8×8 mean-pool blocks (64-d)
    sx  = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy  = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    #   reshape into 8×8 non-overlapping blocks of 32×32 pixels
    block  = 256 // _GRAD_CELLS       # = 32
    grads  = []
    for r in range(_GRAD_CELLS):
        for c in range(_GRAD_CELLS):
            cell = mag[r*block:(r+1)*block, c*block:(c+1)*block]
            grads.append(float(cell.mean()))
    grad_feat = np.array(grads, dtype=np.float32)    # 64-d

    # 5. Concatenate & L2-normalise
    feat = np.concatenate([hist_feat, edge_map, grad_feat]).astype(np.float32)
    norm = np.linalg.norm(feat)
    if norm > 1e-6:
        feat /= norm
    return feat


def _feat_from_path(img_path: str) -> np.ndarray:
    """Read image file → feature. Returns zero vector on failure."""
    zero = np.zeros(_FEAT_DIM, dtype=np.float32)
    if not img_path or not os.path.exists(img_path):
        return zero
    img = cv2.imread(img_path)
    if img is None:
        return zero
    return _extract(img)


def _file_hash(path: str) -> str:
    """Quick hash based on file size + mtime — no full file read needed."""
    try:
        st = os.stat(path)
        return f"{st.st_size}_{int(st.st_mtime)}"
    except Exception:
        return "?"


# ── Per-part representative feature ──────────────────────────────────────────

def _part_feature(part: dict) -> np.ndarray:
    """
    Returns the visual feature for a part.
    Aggregates up to 5 images by averaging their features.
    Caches result keyed by part_no + hash of first image.
    """
    global _cache_dirty

    sys.path.insert(0, ROOT)
    from database.db_manager import get_part_images

    pno  = part['part_no']
    imgs = get_part_images(pno)

    # Collect valid image paths (prefer annotated first)
    paths = []
    for img in sorted(imgs, key=lambda x: -x.get('annotated', 0)):
        p = img.get('image_path', '')
        if p and os.path.exists(p):
            paths.append(p)
        if len(paths) >= 5:
            break

    # Fallback to image_path field
    if not paths:
        fp = part.get('image_path', '') or ''
        if fp and os.path.exists(fp):
            paths = [fp]

    if not paths:
        return np.zeros(_FEAT_DIM, dtype=np.float32)

    # Cache key = part_no + hash of first image
    cache_key = f"{pno}::{_file_hash(paths[0])}"

    if cache_key in _mem_cache:
        return np.array(_mem_cache[cache_key], dtype=np.float32)

    # Compute average feature over up to 5 images
    feats = [_feat_from_path(p) for p in paths]
    valid = [f for f in feats if np.linalg.norm(f) > 1e-6]
    if not valid:
        return np.zeros(_FEAT_DIM, dtype=np.float32)

    avg  = np.mean(valid, axis=0).astype(np.float32)
    norm = np.linalg.norm(avg)
    if norm > 1e-6:
        avg /= norm

    _mem_cache[cache_key] = avg.tolist()
    _cache_dirty = True
    _save_cache()
    return avg


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    da = np.linalg.norm(a)
    db = np.linalg.norm(b)
    if da < 1e-9 or db < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (da * db))


# ── Public API ────────────────────────────────────────────────────────────────

_cache_loaded = False


def find_similar(query_part: dict,
                 query_img_path: str = None,
                 top_n: int = 15) -> list:
    """
    Find parts visually similar to query_part.

    Parameters
    ----------
    query_part     : dict from get_part_by_no()
    query_img_path : optional raw frame path (live camera crop)
                     — if given, used as query image instead of stored images
    top_n          : max results to return

    Returns
    -------
    list of dicts, sorted by score descending:
        {
            'part'  : dict,
            'score' : float  0-1,
            'tier'  : 'exact' | 'similar' | 'slight'
        }
    Only results with tier != None are returned.
    """
    global _cache_loaded
    if not _cache_loaded:
        _load_cache()
        _cache_loaded = True

    sys.path.insert(0, ROOT)
    from database.db_manager import get_all_parts

    # Query feature
    if query_img_path and os.path.exists(query_img_path):
        q_feat = _feat_from_path(query_img_path)
        # If zero (bad image), fall back to stored feature
        if np.linalg.norm(q_feat) < 1e-6:
            q_feat = _part_feature(query_part)
    else:
        q_feat = _part_feature(query_part)

    if np.linalg.norm(q_feat) < 1e-6:
        return []     # no images at all → can't compare

    q_pno     = query_part.get('part_no', '')
    all_parts = get_all_parts()

    results = []
    for p in all_parts:
        if p['part_no'] == q_pno:
            continue

        p_feat = _part_feature(p)
        if np.linalg.norm(p_feat) < 1e-6:
            continue   # this part has no images → skip

        score = _cosine(q_feat, p_feat)

        if score >= TIER_EXACT:
            tier = 'exact'
        elif score >= TIER_SIMILAR:
            tier = 'similar'
        elif score >= TIER_SLIGHT:
            tier = 'slight'
        else:
            tier = None

        if tier is not None:
            results.append({'part': p, 'score': round(score, 3), 'tier': tier})

    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_n]


def feature_from_frame_crop(frame_bgr: np.ndarray,
                             bbox: tuple) -> np.ndarray:
    """
    Extract feature directly from a live camera bounding-box crop.
    bbox = (x1, y1, x2, y2) in frame pixel coords.
    """
    x1, y1, x2, y2 = bbox
    crop = frame_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop.size == 0:
        return np.zeros(_FEAT_DIM, dtype=np.float32)
    return _extract(crop)
