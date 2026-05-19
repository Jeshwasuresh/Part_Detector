"""
Training Pipeline  —  offline-first, high accuracy
====================================================
• Uses pretrained/yolov8s.pt (pre-downloaded) as base — no internet needed
• Offline augmentation for small datasets (<30 images per class)
• Cosine LR, AdamW, mosaic, mixup, copy-paste
• Per-class balanced 80/20 split
• Post-training validation + grade report
"""
import os, sys, shutil, random, yaml, json, cv2
import numpy as np
from datetime import datetime
from pathlib import Path

# ── Fix 1: redirect Ultralytics config away from any stale/missing drive ──────
_CFG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.ultralytics_cfg')
os.makedirs(_CFG_DIR, exist_ok=True)
os.environ['YOLO_CONFIG_DIR'] = _CFG_DIR   # force-override system-level E:\ var

# ── Fix 2: PyTorch 2.6+ changed torch.load default to weights_only=True ───────
#    Ultralytics .pt files contain non-tensor objects, so we force False.
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

ROOT       = os.path.dirname(os.path.abspath(__file__))
DATASET    = os.path.join(ROOT, 'dataset')
MODEL_DIR  = os.path.join(ROOT, 'trained_model')
PRE_DIR    = os.path.join(MODEL_DIR, 'pretrained')
LOG_DIR    = os.path.join(ROOT, 'training_logs')
YAML_PATH  = os.path.join(DATASET, 'dataset.yaml')
AUG_DIR    = os.path.join(ROOT, 'parts_images', '_aug')

MIN_PER_CLASS    = 30
TARGET_PER_CLASS = 60
sys.path.insert(0, ROOT)


# ── Offline augmentation ───────────────────────────────────────────────────────
def _augment(img_path, lbl_path, out_img, out_lbl, n=4):
    img = cv2.imread(img_path)
    if img is None: return []
    H,W = img.shape[:2]
    lines = [l.strip() for l in open(lbl_path) if l.strip()]
    results = []
    for i in range(n):
        aug = img.copy()
        new_lines = list(lines)
        ops = random.sample(['bright','contrast','blur','noise','hue','shadow','fliph'], k=random.randint(2,3))
        for op in ops:
            if op=='bright':
                aug=np.clip(aug.astype(np.int32)+random.uniform(-50,50),0,255).astype(np.uint8)
            elif op=='contrast':
                aug=np.clip(aug.astype(np.float32)*random.uniform(0.6,1.4),0,255).astype(np.uint8)
            elif op=='blur':
                aug=cv2.GaussianBlur(aug,(random.choice([3,5]),)*2,0)
            elif op=='noise':
                aug=np.clip(aug.astype(np.int16)+np.random.randint(-20,20,aug.shape,np.int16),0,255).astype(np.uint8)
            elif op=='hue':
                h=cv2.cvtColor(aug,cv2.COLOR_BGR2HSV).astype(np.int32)
                h[:,:,0]=(h[:,:,0]+random.randint(-15,15))%180
                aug=cv2.cvtColor(h.astype(np.uint8),cv2.COLOR_HSV2BGR)
            elif op=='shadow':
                y1,y2=sorted(random.sample(range(H),2))
                s=aug.copy(); s[y1:y2,:]=(s[y1:y2,:]*0.55).astype(np.uint8); aug=s
            elif op=='fliph':
                aug=cv2.flip(aug,1)
                new_lines=[]
                for ln in lines:
                    p=ln.split()
                    if len(p)==5: p[1]=str(round(1.0-float(p[1]),6)); new_lines.append(' '.join(p))
                    else: new_lines.append(ln)
        base=f"{Path(img_path).stem}_aug{i}"
        oi=os.path.join(out_img,base+'.jpg'); ol=os.path.join(out_lbl,base+'.txt')
        cv2.imwrite(oi,aug,[cv2.IMWRITE_JPEG_QUALITY,92])
        open(ol,'w').write('\n'.join(new_lines))
        results.append((oi,ol))
    return results


def run_training(epochs=80, img_size=640, model_size='s',
                 progress_cb=None, log_cb=None):
    def _p(pct,msg):
        if progress_cb: progress_cb(pct,msg)
    def _l(msg):
        if log_cb: log_cb(msg)
        print(msg)

    os.makedirs(MODEL_DIR,exist_ok=True); os.makedirs(LOG_DIR,exist_ok=True)

    # 1. ultralytics
    _p(2,"Checking ultralytics...")
    try: from ultralytics import YOLO
    except ImportError:
        return {'success':False,'message':'Run offline_setup.py first to install packages.','metrics':{}}

    # 2. Load annotated images
    _p(5,"Loading annotated images...")
    from database.db_manager import get_all_annotated_images, get_all_yolo_classes
    images  = get_all_annotated_images()
    classes = get_all_yolo_classes()
    if not images:
        return {'success':False,'message':(
            'No annotated images found!\n\n'
            'Steps:\n 1. Add Part → set YOLO Class\n'
            ' 2. Images & Annotate tab\n'
            ' 3. Upload 30+ photos\n'
            ' 4. ✏️ Annotate each (draw bounding box)\n'
            ' 5. Come back and Train'),'metrics':{}}
    class_map={c:i for i,c in enumerate(classes)}
    _l(f"Classes ({len(classes)}): {classes}")
    _l(f"Total annotated: {len(images)}")
    _p(8,f"{len(images)} images | {len(classes)} classes")

    # 3. Clear dirs
    for sub in ['images/train','images/val','labels/train','labels/val']:
        d=os.path.join(DATASET,sub); shutil.rmtree(d,ignore_errors=True); os.makedirs(d)

    # 4. Offline augmentation for small classes
    _p(12,"Augmenting small classes...")
    by_cls={}
    for img in images: by_cls.setdefault(img['yolo_class'],[]).append(img)
    os.makedirs(os.path.join(AUG_DIR,'images'),exist_ok=True)
    os.makedirs(os.path.join(AUG_DIR,'labels'),exist_ok=True)
    for cls_name,cls_imgs in by_cls.items():
        if len(cls_imgs)<MIN_PER_CLASS:
            needed=TARGET_PER_CLASS-len(cls_imgs); per=max(1,needed//len(cls_imgs)+1); gen=0
            for item in cls_imgs:
                if gen>=needed: break
                if not os.path.exists(item['image_path']): continue
                if not os.path.exists(item['label_path']): continue
                for ni,nl in _augment(item['image_path'],item['label_path'],
                                      os.path.join(AUG_DIR,'images'),
                                      os.path.join(AUG_DIR,'labels'),n=per):
                    by_cls[cls_name].append({'image_path':ni,'label_path':nl,'yolo_class':cls_name})
                    gen+=1
                    if gen>=needed: break
            _l(f"  {cls_name}: {len(cls_imgs)-gen} orig + {gen} augmented = {len(by_cls[cls_name])}")
        else:
            _l(f"  {cls_name}: {len(cls_imgs)} images (no augmentation needed)")

    # 5. 80/20 split
    _p(20,"Splitting 80/20 per class...")
    train_list=[]; val_list=[]
    for cls_name,cls_imgs in by_cls.items():
        random.shuffle(cls_imgs); nv=max(2,int(len(cls_imgs)*0.2))
        val_list+=cls_imgs[:nv]; train_list+=cls_imgs[nv:]
        _l(f"  {cls_name}: {len(cls_imgs)-nv} train / {nv} val")

    # 6. Copy to dataset dirs
    _p(28,"Copying images & labels...")
    def copy_set(lst,split):
        ok=0
        for item in lst:
            src_img=item['image_path']; src_lbl=item['label_path']; yc=item['yolo_class']
            if not os.path.exists(src_img): continue
            if not os.path.exists(src_lbl): continue
            base=os.path.splitext(os.path.basename(src_img))[0]
            di=os.path.join(DATASET,f'images/{split}/{base}.jpg')
            dl=os.path.join(DATASET,f'labels/{split}/{base}.txt')
            img=cv2.imread(src_img)
            if img is None: continue
            cv2.imwrite(di,img,[cv2.IMWRITE_JPEG_QUALITY,95])
            ci=class_map.get(yc,0)
            lns=[f"{ci} "+" ".join(ln.split()[1:]) for ln in open(src_lbl) if len(ln.split())==5]
            if lns: open(dl,'w').write('\n'.join(lns)); ok+=1
        return ok
    n_train=copy_set(train_list,'train'); n_val=copy_set(val_list,'val')
    _l(f"Dataset: {n_train} train / {n_val} val")
    _p(32,f"Dataset ready: {n_train} train / {n_val} val")
    if n_train==0:
        return {'success':False,'message':'No valid image/label pairs found.','metrics':{}}

    # 7. YAML
    yaml_data={'path':DATASET,'train':'images/train','val':'images/val','nc':len(classes),'names':classes}
    open(YAML_PATH,'w').write(yaml.dump(yaml_data))

    # 8. Find base model (offline-first)
    base_model_path=None
    for name in [f'yolov8{model_size}.pt', 'yolov8s.pt', 'yolov8n.pt']:
        p=os.path.join(PRE_DIR,name)
        if os.path.exists(p): base_model_path=p; break
    if not base_model_path:
        # Try ultralytics auto-download as last resort
        base_model_path=f'yolov8{model_size}.pt'
        _l(f"WARNING: No pretrained weights in {PRE_DIR} — trying auto-download (needs internet)")
    _l(f"Base model: {base_model_path}")

    run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _p(35,f"Training YOLOv8{model_size} — {epochs} epochs...")
    _l("="*55); _l(f"TRAINING STARTED  {datetime.now().strftime('%H:%M:%S')}")
    _l(f"Model: {base_model_path}  Epochs: {epochs}  ImgSize: {img_size}")
    _l("="*55)

    try:
        model=YOLO(base_model_path)
        def on_epoch(trainer):
            ep=trainer.epoch+1; tot=trainer.epochs
            pct=35+int((ep/tot)*52); msg=f"Epoch {ep}/{tot}"
            if hasattr(trainer,'metrics') and trainer.metrics:
                m50=trainer.metrics.get('metrics/mAP50(B)',0) or 0
                pr =trainer.metrics.get('metrics/precision(B)',0) or 0
                rc =trainer.metrics.get('metrics/recall(B)',0) or 0
                try: msg+=f"  mAP50={float(m50):.3f}  P={float(pr):.3f}  R={float(rc):.3f}"
                except: pass
            _p(pct,msg); _l(msg)
        model.add_callback('on_train_epoch_end',on_epoch)

        model.train(
            data=YAML_PATH, epochs=epochs, imgsz=img_size,
            project=MODEL_DIR, name=run_name, exist_ok=True,
            verbose=False, patience=20, batch=16,
            optimizer='AdamW', lr0=0.001, lrf=0.01,
            momentum=0.937, weight_decay=0.0005,
            warmup_epochs=5, cos_lr=True,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            degrees=15.0, translate=0.1, scale=0.6,
            shear=5.0, flipud=0.1, fliplr=0.5,
            mosaic=1.0, mixup=0.1, copy_paste=0.1,
            close_mosaic=10,
            box=7.5, cls=0.5, dfl=1.5,
        )

        _p(88,"Saving best model...")
        best_src=os.path.join(MODEL_DIR,run_name,'weights','best.pt')
        best_dst=os.path.join(MODEL_DIR,'best.pt')
        if os.path.exists(best_src): shutil.copy2(best_src,best_dst); _l(f"✅ best.pt → {best_dst}")
        else:
            last=os.path.join(MODEL_DIR,run_name,'weights','last.pt')
            if os.path.exists(last): shutil.copy2(last,best_dst); _l("✅ last.pt saved")
            else: return {'success':False,'message':'No weights saved — training may have failed.','metrics':{}}

        _p(92,"Reading metrics...")
        metrics={}
        try:
            import csv as _csv
            rc=os.path.join(MODEL_DIR,run_name,'results.csv')
            if os.path.exists(rc):
                rows=list(_csv.DictReader(open(rc)))
                if rows: metrics={k.strip():(v.strip() if v else '0') for k,v in rows[-1].items()}
        except Exception as e: _l(f"Metrics error: {e}")

        def _m(k): 
            try: return float(metrics.get(k,0) or 0)
            except: return 0.0
        mAP50=_m('metrics/mAP50(B)'); mAP5095=_m('metrics/mAP50-95(B)')
        prec=_m('metrics/precision(B)'); rec=_m('metrics/recall(B)')

        _p(95,"Running validation...")
        try: YOLO(best_dst).val(data=YAML_PATH,verbose=False,conf=0.50,iou=0.45)
        except Exception as e: _l(f"Validation skipped: {e}")

        grade=('A' if mAP50>=0.85 else 'B' if mAP50>=0.70 else 'C' if mAP50>=0.50 else 'D')
        advice={'A':"✅ Excellent! Ready for production.",
                'B':"✅ Good. Add more varied images to improve.",
                'C':"⚠️  Moderate. Add 20+ more images per part & retrain.",
                'D':("❌ Low accuracy.\n"
                     "  • Need 50+ images per part\n"
                     "  • Plain background\n"
                     "  • Tight bounding boxes\n"
                     "  • More angle variety")}[grade]

        json.dump({'run':run_name,'model':base_model_path,'epochs':epochs,
                   'classes':classes,'n_train':n_train,'n_val':n_val,
                   'mAP50':mAP50,'precision':prec,'recall':rec,'grade':grade},
                  open(os.path.join(LOG_DIR,f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"),'w'),indent=2)

        _p(100,f"✅ Done! mAP50={mAP50:.3f}  Grade={grade}")
        _l("="*55); _l(f"Grade: {grade}  mAP50={mAP50:.3f}  P={prec:.3f}  R={rec:.3f}")
        _l(advice); _l("="*55)

        return {'success':True,'metrics':metrics,'model_path':best_dst,
                'mAP50':mAP50,'grade':grade,
                'message':(f"Training Complete!\n\nGrade      : {grade}\n"
                           f"mAP50      : {mAP50:.3f}  (target ≥ 0.85)\n"
                           f"mAP50-95   : {mAP5095:.3f}\n"
                           f"Precision  : {prec:.3f}\n"
                           f"Recall     : {rec:.3f}\n\n"
                           f"Model      : YOLOv8{model_size}\n"
                           f"Classes    : {classes}\n"
                           f"Train imgs : {n_train}  (incl. augmented)\n"
                           f"Val imgs   : {n_val}\n\n{advice}")}
    except Exception as e:
        import traceback; _l(f"ERROR:\n{traceback.format_exc()}")
        return {'success':False,'message':f'Training failed:\n{str(e)}','metrics':{}}
