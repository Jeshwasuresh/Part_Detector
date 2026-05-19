"""
AutoPartDetector GUI — Tablet Edition
======================================
• Virtual on-screen keyboard pops up on any Entry tap
• Swipe anywhere on page body to scroll (not just scrollbar thumb)
• All buttons min 48px tall for finger touch
• Fully offline after first setup
• 3 tabs in Add Part: Info | Images+Annotate | Upload YOLO .txt
• Right panel: Exact / Similar / Slightly Similar (visual only)
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2, os, sys, shutil, threading, time, csv, json, zipfile
from datetime import datetime
from PIL import Image, ImageTk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from database.db_manager import (
    init_db, get_all_parts, get_part_by_no, get_part_by_yolo_class,
    add_part, delete_part, search_parts,
    add_part_image, get_part_images, update_part_image_label,
    delete_part_image, get_all_yolo_classes, get_all_annotated_images,
    count_annotated, log_detection, get_history, clear_history, PARTS_IMG_DIR
)
from detection.detector import PartDetector
from tablet_utils import (
    VirtualKeyboard, attach_kb, attach_kb_all,
    make_touch_scrollable, TouchScroll,
    setup_global_mousewheel, register_scroll_canvas
)

# ── Palette ───────────────────────────────────────────────────────────────────
BG     = "#0d1b2e"
CARD   = "#112240"
HDR    = "#1565c0"
PANEL  = "#0a1628"
BORDER = "#1e3a5f"
TEXT   = "#e0e8f0"
MUTED  = "#6a8aaa"
GREEN  = "#00e676"
CYAN   = "#00bcd4"
ORANGE = "#ff9800"
RED    = "#f44336"
YELLOW = "#ffeb3b"
BLU    = "#1976d2"
DRED   = "#c62828"
TEAL   = "#00838f"
GREY   = "#37474f"
PURPLE = "#9c27b0"
C_EXACT  = "#00e676"
C_SIM    = "#29b6f6"
C_SLIGHT = "#ff9800"

# Tablet-size fonts (larger for touch)
FT   = ("Arial", 17, "bold")
FH2  = ("Arial", 13, "bold")
FH3  = ("Arial", 11, "bold")
FB   = ("Arial", 11)
FSM  = ("Arial", 10)
FM   = ("Courier New", 10)

SCREENS_DIR = os.path.join(ROOT, 'screenshots')
os.makedirs(SCREENS_DIR,   exist_ok=True)
os.makedirs(PARTS_IMG_DIR, exist_ok=True)


def jcol(j):
    j = (j or '').lower()
    if 'scrap'  in j: return RED
    if 'ok'     in j: return GREEN
    if 'rework' in j: return ORANGE
    if 'return' in j: return PURPLE
    return CYAN


def TB(parent, text, cmd, bg=BLU, fg='white', font=FH3, **kw):
    """Touch Button — minimum 48 px tall for finger touch."""
    return tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg, font=font,
                     relief='flat', cursor='hand2', pady=10,
                     activebackground=bg, activeforeground=fg, **kw)


def entry(parent, var, root, ph='', width=0, **kw):
    """Entry with virtual keyboard auto-attach."""
    e = tk.Entry(parent, textvariable=var, bg=CARD, fg=MUTED,
                 insertbackground=TEXT, font=FB, relief='flat', bd=0,
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=CYAN, **kw)
    if width:
        e.config(width=width)
    if ph:
        e.insert(0, ph)
        e.bind('<FocusIn>',
               lambda ev, _e=e, _p=ph:
               (_e.delete(0, 'end'), _e.config(fg=TEXT))
               if _e.get() == _p else None)
        e.bind('<FocusOut>',
               lambda ev, _e=e, _p=ph:
               (_e.insert(0, _p), _e.config(fg=MUTED))
               if not _e.get() else None)
    attach_kb(root, e)
    return e


# ══════════════════════════════════════════════════════════════════════════════
#  BOUNDING BOX ANNOTATOR
# ══════════════════════════════════════════════════════════════════════════════
_HANDLE_R = 7

class AnnotatorWindow(tk.Toplevel):
    def __init__(self, parent, img_record, yolo_class, on_save_cb=None,
                 img_list=None, img_index=None):
        super().__init__(parent)
        self.img_rec    = img_record
        self.yolo_class = yolo_class
        self.on_save_cb = on_save_cb
        self._img_list  = img_list  if img_list  is not None else [img_record]
        self._img_index = img_index if img_index is not None else 0
        total = len(self._img_list)
        pos   = self._img_index + 1
        fname = os.path.basename(img_record['image_path'])
        self.title(f"Annotate  [{pos} / {total}]  --  {fname}")
        self.configure(bg=BG)
        try: self.state('zoomed')
        except Exception:
            try: self.attributes('-zoomed', True)
            except Exception: self.geometry('1366x768')
        self._boxes  = []
        self._orig   = None
        self._scale  = 1.0
        self._ox = self._oy = 0
        self._cw = 900; self._ch = 600
        self._mode      = 'idle'
        self._start     = None
        self._cur       = None
        self._drag_idx  = None
        self._drag_corn = None
        self._drag_orig = None
        self._build(); self._load_img(); self._load_existing()
        self.grab_set()

    def _build(self):
        total = len(self._img_list)
        pos   = self._img_index + 1
        hdr = tk.Frame(self, bg=HDR, pady=10)
        hdr.pack(side='top', fill='x')
        tk.Label(hdr, text="Drag empty = new box  |  Drag inside box = move  |  Drag corner = resize  |  Right-click = delete",
                 bg=HDR, fg='white', font=FH2).pack(side='left', padx=14)
        tk.Label(hdr, text=f"Image {pos} of {total}",
                 bg=HDR, fg=YELLOW, font=('Arial',12,'bold')).pack(side='right', padx=16)
        # FOOTER
        footer = tk.Frame(self, bg='#0a1628', pady=10)
        footer.pack(side='bottom', fill='x')
        TB(footer, "Save & Next", self._save,
           bg=GREEN, fg='#000', font=('Arial', 14, 'bold'),
           padx=24).pack(side='left', padx=12, pady=2)
        TB(footer, "Skip ->", self._skip,
           bg=ORANGE, fg='#000', font=FH2,
           padx=16).pack(side='left', padx=4)
        TB(footer, "X Close", self.destroy,
           bg=DRED, fg='white', font=FH2,
           padx=16).pack(side='left', padx=4)
        self.st = tk.Label(footer, text="Draw boxes, then Save & Next",
                           bg='#0a1628', fg=MUTED, font=FSM)
        self.st.pack(side='right', padx=16)
        # BODY
        body = tk.Frame(self, bg=BG)
        body.pack(side='top', fill='both', expand=True)
        self.cv = tk.Canvas(body, bg='#050e1a', cursor='crosshair',
                            highlightthickness=0)
        self.cv.pack(side='left', fill='both', expand=True)
        for ev, fn in [('<ButtonPress-1>',  self._dn),
                       ('<B1-Motion>',       self._mv),
                       ('<ButtonRelease-1>', self._up),
                       ('<ButtonPress-3>',   self._rc),
                       ('<Configure>',       self._resize)]:
            self.cv.bind(ev, fn)
        rp = tk.Frame(body, bg=CARD, width=240, padx=12, pady=12)
        rp.pack(side='right', fill='y')
        rp.pack_propagate(False)
        tk.Label(rp, text="YOLO Class:", bg=CARD, fg=MUTED, font=FSM).pack(anchor='w')
        tk.Label(rp, text=self.yolo_class, bg=CARD, fg=YELLOW,
                 font=('Arial', 12, 'bold')).pack(anchor='w', pady=(0, 10))
        self.cnt = tk.Label(rp, text="Boxes: 0", bg=CARD, fg=TEXT, font=FH3)
        self.cnt.pack(anchor='w')
        self.lb = tk.Listbox(rp, bg=PANEL, fg=TEXT, font=FSM,
                             height=8, selectbackground=BLU,
                             relief='flat', bd=0)
        self.lb.pack(fill='x', pady=4)
        self.lb.bind('<<ListboxSelect>>', self._hl)
        TB(rp, "Delete Selected", self._del_sel, bg=DRED, font=FSM).pack(fill='x', pady=2)
        TB(rp, "Clear All Boxes", self._clr, bg=GREY, font=FSM).pack(fill='x', pady=2)
        tk.Frame(rp, bg=BORDER, height=1).pack(fill='x', pady=10)
        tk.Frame(rp, bg=BORDER, height=1).pack(fill='x', pady=10)
        TB(rp, "Save & Next", self._save,
           bg=GREEN, fg='#000',
           font=('Arial', 11, 'bold')).pack(fill='x', pady=4)

    def _load_img(self):
        p = self.img_rec['image_path']
        if not os.path.exists(p): self.destroy(); return
        self._orig = Image.open(p).convert('RGB'); self._redraw()

    def _load_existing(self):
        lp = self.img_rec.get('label_path','')
        if not lp or not os.path.exists(lp): return
        W,H = self._orig.size
        for line in open(lp):
            pts = line.strip().split()
            if len(pts)==5:
                _,cx,cy,bw,bh = map(float,pts)
                self._boxes.append([(cx-bw/2)*W,(cy-bh/2)*H,(cx+bw/2)*W,(cy+bh/2)*H])
        self._redraw(); self._upd()

    def _resize(self,e): self._cw=e.width; self._ch=e.height; self._redraw()

    def _redraw(self):
        if not self._orig: return
        W,H = self._orig.size
        sc = min(self._cw/W, self._ch/H)
        nw,nh = int(W*sc),int(H*sc)
        self._scale=sc; self._ox=(self._cw-nw)//2; self._oy=(self._ch-nh)//2
        img = self._orig.resize((nw,nh),Image.BILINEAR)
        self._photo = ImageTk.PhotoImage(img)
        self.cv.delete('all')
        self.cv.create_image(self._ox,self._oy,anchor='nw',image=self._photo)
        for i,box in enumerate(self._boxes): self._draw_box(box,i)

    def _i2c(self,x,y): return x*self._scale+self._ox, y*self._scale+self._oy
    def _c2i(self,cx,cy): return (cx-self._ox)/self._scale,(cy-self._oy)/self._scale

    def _draw_box(self, box, idx, hi=False):
        x1,y1,x2,y2 = box
        cx1,cy1 = self._i2c(x1,y1)
        cx2,cy2 = self._i2c(x2,y2)
        col  = YELLOW if hi else GREEN
        tag  = f"b{idx}"
        r    = _HANDLE_R
        lw   = 3 if hi else 2

        # Main rectangle
        self.cv.create_rectangle(cx1, cy1, cx2, cy2,
                                  outline=col, width=lw, tag=tag)
        # Label
        self.cv.create_text(cx1+5, cy1+3,
                             text=f"{self.yolo_class}#{idx+1}",
                             anchor='nw', fill=col,
                             font=("Arial", 9, "bold"), tag=tag)
        # ── Four corner handles ───────────────────────────────────────────
        h_col  = '#ff4081' if hi else '#ffeb3b'
        h_out  = '#fff' if hi else col
        for (hx, hy, cname) in [
            (cx1, cy1, 'tl'), (cx2, cy1, 'tr'),
            (cx1, cy2, 'bl'), (cx2, cy2, 'br'),
        ]:
            htag = f"h{idx}_{cname}"
            self.cv.create_rectangle(
                hx-r, hy-r, hx+r, hy+r,
                fill=h_col, outline=h_out, width=2,
                tag=(tag, htag))

    # ── Helper: find corner handle or box under canvas point ─────────────────
    def _hit_test(self, cx, cy):
        """Return ('corner', idx, cname) | ('box', idx) | ('empty', -1, '')."""
        r = _HANDLE_R + 2
        for i, box in enumerate(self._boxes):
            bx1,by1,bx2,by2 = box
            # canvas coords of this box
            ccx1,ccy1 = self._i2c(bx1,by1)
            ccx2,ccy2 = self._i2c(bx2,by2)
            corners = [('tl',ccx1,ccy1),('tr',ccx2,ccy1),
                       ('bl',ccx1,ccy2),('br',ccx2,ccy2)]
            for cname,hx,hy in corners:
                if abs(cx-hx) <= r and abs(cy-hy) <= r:
                    return ('corner', i, cname)
        # Check body (check in reverse so top-drawn box wins)
        for i in range(len(self._boxes)-1, -1, -1):
            bx1,by1,bx2,by2 = self._boxes[i]
            ccx1,ccy1 = self._i2c(bx1,by1)
            ccx2,ccy2 = self._i2c(bx2,by2)
            if ccx1 <= cx <= ccx2 and ccy1 <= cy <= ccy2:
                return ('body', i, '')
        return ('empty', -1, '')

    def _dn(self, e):
        hit, idx, cname = self._hit_test(e.x, e.y)
        self._start    = (e.x, e.y)
        self._cur      = None
        self._drag_idx  = idx
        self._drag_corn = cname
        if hit == 'corner':
            self._mode     = 'resize'
            self._drag_orig = list(self._boxes[idx])
            self.cv.config(cursor='sizing')
        elif hit == 'body':
            self._mode     = 'move'
            self._drag_orig = list(self._boxes[idx])
            self.cv.config(cursor='fleur')
        else:
            self._mode = 'draw'
            self.cv.config(cursor='crosshair')

    def _mv(self, e):
        if self._start is None: return
        sx, sy = self._start
        dx = e.x - sx;  dy = e.y - sy

        if self._mode == 'resize' and self._drag_idx is not None:
            b  = list(self._drag_orig)
            ix, iy = self._c2i(e.x, e.y)
            W, H = self._orig.size
            ix = max(0, min(ix, W));  iy = max(0, min(iy, H))
            cn = self._drag_corn
            if 'l' in cn: b[0] = ix
            else:         b[2] = ix
            if 't' in cn: b[1] = iy
            else:         b[3] = iy
            # keep valid
            b[0],b[2] = min(b[0],b[2]), max(b[0],b[2])
            b[1],b[3] = min(b[1],b[3]), max(b[1],b[3])
            self._boxes[self._drag_idx] = b
            self._redraw(); self._upd()

        elif self._mode == 'move' and self._drag_idx is not None:
            b  = list(self._drag_orig)
            W, H = self._orig.size
            ddx, ddy = dx/self._scale, dy/self._scale
            bw = b[2]-b[0]; bh = b[3]-b[1]
            nx1 = max(0, min(b[0]+ddx, W-bw))
            ny1 = max(0, min(b[1]+ddy, H-bh))
            self._boxes[self._drag_idx] = [nx1, ny1, nx1+bw, ny1+bh]
            self._redraw(); self._upd()

        elif self._mode == 'draw':
            if self._cur: self.cv.delete(self._cur)
            self._cur = self.cv.create_rectangle(
                sx, sy, e.x, e.y,
                outline=YELLOW, width=2, dash=(4, 2))

    def _up(self, e):
        if self._start is None: return
        sx, sy = self._start
        self.cv.config(cursor='crosshair')

        if self._mode == 'draw':
            if self._cur:
                self.cv.delete(self._cur)
                self._cur = None
            # Only create box if dragged far enough
            if abs(e.x-sx) >= 20 and abs(e.y-sy) >= 20:
                ix1,iy1 = self._c2i(sx, sy)
                ix2,iy2 = self._c2i(e.x, e.y)
                W,H = self._orig.size
                x1 = max(0, min(ix1, ix2))
                y1 = max(0, min(iy1, iy2))
                x2 = min(W, max(ix1, ix2))
                y2 = min(H, max(iy1, iy2))
                if (x2-x1) >= 5 and (y2-y1) >= 5:
                    self._boxes.append([x1, y1, x2, y2])
                    self._redraw(); self._upd()

        # resize / move already applied live — just clean up
        self._mode      = 'idle'
        self._start     = None
        self._drag_idx  = None
        self._drag_corn = None
        self._drag_orig = None
    def _rc(self,e):
        ix,iy=self._c2i(e.x,e.y)
        for i,(x1,y1,x2,y2) in enumerate(self._boxes):
            if x1<=ix<=x2 and y1<=iy<=y2: self._boxes.pop(i); self._redraw(); self._upd(); return
    def _upd(self):
        self.lb.delete(0,'end')
        for i,(x1,y1,x2,y2) in enumerate(self._boxes):
            self.lb.insert('end',f"#{i+1} ({int(x1)},{int(y1)})→({int(x2)},{int(y2)})")
        self.cnt.config(text=f"Boxes: {len(self._boxes)}")
    def _hl(self,e):
        sel=self.lb.curselection()
        if not sel: return
        self._redraw(); self._draw_box(self._boxes[sel[0]],sel[0],hi=True)
    def _del_sel(self):
        sel=self.lb.curselection()
        if sel: self._boxes.pop(sel[0]); self._redraw(); self._upd()
    def _clr(self): self._boxes.clear(); self._redraw(); self._upd()

    def _save(self):
        if not self._boxes:
            messagebox.showwarning("No Boxes",
                "Draw at least one bounding box around the part first.\n\n"
                "Left-drag on the image to draw a box.")
            return
        W, H = self._orig.size
        ldir  = os.path.join(ROOT, 'dataset', 'labels_raw')
        os.makedirs(ldir, exist_ok=True)
        base  = os.path.splitext(os.path.basename(self.img_rec['image_path']))[0]
        lpath = os.path.join(ldir, base + '.txt')
        lines = []
        for x1, y1, x2, y2 in self._boxes:
            cx = ((x1+x2)/2)/W; cy = ((y1+y2)/2)/H
            bw = (x2-x1)/W;     bh = (y2-y1)/H
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        open(lpath, 'w').write('\n'.join(lines))
        update_part_image_label(self.img_rec['id'], lpath, annotated=1)
        try:
            from similarity_engine import invalidate
            invalidate(self.img_rec.get('part_no', ''))
        except Exception:
            pass
        self.st.config(text=f"✅  Saved {len(self._boxes)} box(es)!", fg=GREEN)
        if self.on_save_cb:
            self.on_save_cb()
        self.after(600, self._next_image)

    def _skip(self):
        self._next_image()

    def _next_image(self):
        next_idx = self._img_index + 1
        if next_idx >= len(self._img_list):
            self.st.config(text="✅  All images annotated!", fg=GREEN)
            messagebox.showinfo("Done", "All images processed!", parent=self)
            self.destroy()
            return
        next_rec = self._img_list[next_idx]
        cb, cls, lst = self.on_save_cb, self.yolo_class, self._img_list
        self.destroy()
        AnnotatorWindow(self.master, next_rec, cls, on_save_cb=cb, img_list=lst, img_index=next_idx)

# ══════════════════════════════════════════════════════════════════════════════
#  YOLO .TXT UPLOAD PANEL
# ══════════════════════════════════════════════════════════════════════════════
class YoloUploadPanel(tk.Frame):
    def __init__(self, parent, get_pno_fn, get_cls_fn, root_win):
        super().__init__(parent, bg=BG)
        self._get_pno = get_pno_fn
        self._get_cls = get_cls_fn
        self._root    = root_win
        self._build()

    def _build(self):
        _, body = make_touch_scrollable(self, BG)
        pad = tk.Frame(body, bg=BG); pad.pack(fill='x', padx=18, pady=14)

        tk.Label(pad,text="📂  Upload YOLO .txt Annotations",bg=BG,fg=CYAN,font=FH2).pack(anchor='w')
        tk.Label(pad,text="Import existing YOLO labels — no need to draw boxes manually.",
                 bg=BG,fg=MUTED,font=FSM).pack(anchor='w',pady=(2,14))

        # Mode A
        fA=tk.Frame(pad,bg=CARD,padx=16,pady=14); fA.pack(fill='x',pady=(0,10))
        tk.Label(fA,text="📄  Mode A — Select Images + .txt Files Together",bg=CARD,fg=YELLOW,font=FH3).pack(anchor='w')
        tk.Label(fA,text=("Select image files AND their YOLO .txt files together.\n"
                          "Same filename = auto-matched:\n"
                          "     part1.jpg  ←→  part1.txt"),
                 bg=CARD,fg=MUTED,font=FSM,justify='left').pack(anchor='w',pady=6)
        TB(fA,"📁  Select Images + .txt Files",self._import_pairs,bg=BLU,fg='white',font=FB,padx=16).pack(anchor='w')

        tk.Frame(pad,bg=BORDER,height=1).pack(fill='x',pady=10)

        # Mode B
        fB=tk.Frame(pad,bg=CARD,padx=16,pady=14); fB.pack(fill='x',pady=(0,10))
        tk.Label(fB,text="📦  Mode B — Upload a ZIP File",bg=CARD,fg=YELLOW,font=FH3).pack(anchor='w')
        tk.Label(fB,text=("ZIP can have any structure — importer finds all\n"
                          "image+label pairs automatically:\n\n"
                          "  mydata.zip/\n"
                          "    images/   part1.jpg  part2.jpg\n"
                          "    labels/   part1.txt  part2.txt"),
                 bg=CARD,fg=MUTED,font=FSM,justify='left').pack(anchor='w',pady=6)
        TB(fB,"📦  Select ZIP File",self._import_zip,bg=TEAL,fg='white',font=FB,padx=16).pack(anchor='w')

        tk.Frame(pad,bg=BORDER,height=1).pack(fill='x',pady=10)

        # Reference
        ref=tk.Frame(pad,bg='#0a1f38',padx=14,pady=12); ref.pack(fill='x')
        tk.Label(ref,text="📖  YOLO .txt Format",bg='#0a1f38',fg=YELLOW,font=FH3).pack(anchor='w')
        tk.Label(ref,text=("One line per box:\n\n"
                           "   class_id   cx   cy   width   height\n\n"
                           "  All values 0.0–1.0 (normalised)\n"
                           "  class_id = 0 (remapped at train time)\n\n"
                           "Example:  0  0.512  0.433  0.310  0.275\n\n"
                           "Free tools: labelImg · Roboflow · CVAT · MakeSense.ai"),
                 bg='#0a1f38',fg=MUTED,font=FSM,justify='left').pack(anchor='w',pady=4)

        tk.Label(pad,text="Import Log:",bg=BG,fg=MUTED,font=FSM).pack(anchor='w',pady=(14,2))
        self.log_box=tk.Text(pad,bg=PANEL,fg=GREEN,font=FM,height=8,relief='flat',state='disabled')
        self.log_box.pack(fill='x')
        TB(pad,"🗑 Clear Log",
           lambda:(self.log_box.config(state='normal'),self.log_box.delete(1.0,'end'),self.log_box.config(state='disabled')),
           bg=GREY,fg='white',font=FSM,padx=10).pack(anchor='e',pady=4)

    def _log(self,msg):
        self.log_box.config(state='normal'); self.log_box.insert('end',msg+'\n')
        self.log_box.see('end'); self.log_box.config(state='disabled')

    def _check(self):
        pno=self._get_pno(); cls=self._get_cls()
        if not pno: messagebox.showwarning("Save Part First","Save Part Information tab first."); return None,None
        if not cls: messagebox.showwarning("YOLO Class Missing","Set YOLO Class in Part Info tab first."); return None,None
        return pno,cls

    def _validate_txt(self,path):
        valid=[]
        try:
            for raw in open(path):
                pts=raw.strip().split()
                if len(pts)!=5: continue
                try:
                    vals=list(map(float,pts))
                    if all(0.0<=v<=1.0 for v in vals[1:]): valid.append(raw.strip())
                except ValueError: continue
        except Exception: pass
        return valid

    def _do_import(self,img_path,txt_path,pno):
        if not os.path.exists(img_path): self._log(f"  ❌ Missing: {os.path.basename(img_path)}"); return False
        lines=self._validate_txt(txt_path)
        if not lines: self._log(f"  ❌ No valid YOLO lines: {os.path.basename(txt_path)}"); return False
        ts=datetime.now().strftime('%Y%m%d%H%M%S%f')
        ext=os.path.splitext(img_path)[1].lower() or '.jpg'
        dst_img=os.path.join(PARTS_IMG_DIR,f"{pno}_{ts}{ext}")
        shutil.copy2(img_path,dst_img)
        ldir=os.path.join(ROOT,'dataset','labels_raw'); os.makedirs(ldir,exist_ok=True)
        dst_lbl=os.path.join(ldir,f"{pno}_{ts}.txt")
        forced=[]
        for line in lines: pts=line.split(); pts[0]='0'; forced.append(' '.join(pts))
        open(dst_lbl,'w').write('\n'.join(forced))
        add_part_image(pno,dst_img,dst_lbl,annotated=1,source='yolo_import')
        return True

    def _import_pairs(self):
        pno,cls=self._check()
        if not pno: return
        paths=filedialog.askopenfilenames(title="Select image files AND .txt files together",
            filetypes=[("Images & Labels","*.jpg *.jpeg *.png *.bmp *.txt"),("All","*.*")])
        if not paths: return
        imgs={os.path.splitext(p)[0]:p for p in paths if not p.endswith('.txt')}
        txts={os.path.splitext(p)[0]:p for p in paths if p.endswith('.txt')}
        ok=skip=0; self._log(f"\n── Importing for [{pno}] ──")
        for stem,img_p in imgs.items():
            txt_p=txts.get(stem)
            if not txt_p: self._log(f"  ⚠  No .txt for {os.path.basename(img_p)} — skipped"); skip+=1; continue
            if self._do_import(img_p,txt_p,pno): self._log(f"  ✅  {os.path.basename(img_p)}"); ok+=1
            else: skip+=1
        self._finish(pno,ok,skip)

    def _import_zip(self):
        pno,cls=self._check()
        if not pno: return
        zpath=filedialog.askopenfilename(title="Select ZIP dataset",
            filetypes=[("ZIP","*.zip"),("All","*.*")])
        if not zpath: return
        tmp=os.path.join(ROOT,'_zip_tmp'); shutil.rmtree(tmp,ignore_errors=True); os.makedirs(tmp)
        self._log(f"\n── Unzipping {os.path.basename(zpath)} ──")
        try:
            with zipfile.ZipFile(zpath) as z: z.extractall(tmp)
        except Exception as e: self._log(f"  ❌ ZIP error: {e}"); return
        img_map={}; lbl_map={}
        for root,_,files in os.walk(tmp):
            for fn in files:
                fp=os.path.join(root,fn); stem=os.path.splitext(fn)[0]
                ext=os.path.splitext(fn)[1].lower()
                if ext in {'.jpg','.jpeg','.png','.bmp'}: img_map[stem]=fp
                elif ext=='.txt': lbl_map[stem]=fp
        ok=skip=0
        for stem,img_p in img_map.items():
            lbl_p=lbl_map.get(stem)
            if not lbl_p: self._log(f"  ⚠  No .txt for {os.path.basename(img_p)}"); skip+=1; continue
            if self._do_import(img_p,lbl_p,pno): self._log(f"  ✅  {os.path.basename(img_p)}"); ok+=1
            else: skip+=1
        shutil.rmtree(tmp,ignore_errors=True)
        self._finish(pno,ok,skip)

    def _finish(self,pno,ok,skip):
        try:
            from similarity_engine import invalidate; invalidate(pno)
        except Exception: pass
        self._log(f"Done: ✅ {ok} imported   ⚠ {skip} skipped")
        messagebox.showinfo("Import Complete",
            f"Imported {ok} annotated image(s).\nSkipped {skip}.\n\n"
            "View them in the  📷 Images & Annotate  tab.")


# ══════════════════════════════════════════════════════════════════════════════
#  DETECTED PARTS PANEL
# ══════════════════════════════════════════════════════════════════════════════
class SimilarityPanel(tk.Frame):
    """Accumulates every part ever seen by the camera.
    Clicking any row instantly loads that part into Part Information."""
    def __init__(self, parent, on_select_cb):
        super().__init__(parent, bg=BG)
        self.on_select   = on_select_cb
        self._history    = []        # confirmed {part, conf, label}
        self._seen_pnos  = set()     # pnos that made it into history
        self._live_pnos  = set()     # pnos currently in camera (in history)
        self._candidates = {}        # pno -> (first_seen_time, part, conf) — not yet confirmed
        self._cards      = []
        self.CONFIRM_SEC = 1.5       # seconds of continuous detection required
        self._build()

    # ── Build UI ───────────────────────────────────────────────────────────────
    def _build(self):
        # Header
        hdr = tk.Frame(self, bg=HDR, pady=8); hdr.pack(fill='x')
        tk.Label(hdr, text="🔍  DETECTED PARTS",
                 bg=HDR, fg='white', font=FH2).pack(side='left', padx=12)
        TB(hdr, "🔄 Reset", self._reset,
           bg='#c62828', fg='white', font=FSM, padx=10
           ).pack(side='right', padx=10)

        # Status bar
        self._status = tk.Label(self, bg=PANEL, fg=MUTED, font=FSM,
                                text="Start camera — parts accumulate here",
                                anchor='w', padx=10, pady=5)
        self._status.pack(fill='x')

        # Scrollable card list
        self._canvas, self._body = make_touch_scrollable(self, BG)

        # Empty placeholder
        self._empty_lbl = tk.Label(
            self._body, bg=BG, fg=MUTED, font=FB,
            text="No parts detected yet.\nShow a part to the camera.",
            pady=30, justify='center')
        self._empty_lbl.pack()

    # ── Public API ─────────────────────────────────────────────────────────────
    def clear(self):
        """Camera stopped — keep history, mark all as not-live, clear candidates."""
        self._live_pnos.clear()
        self._candidates.clear()
        self._redraw()

    def update_results(self, *a, **kw): pass  # backward compat no-op

    def update_detections(self, overlay):
        import time as _t
        now      = _t.time()
        now_pnos = set()    # all pnos visible this frame
        changed  = False

        for item in overlay:
            part = item.get('part')
            if not part: continue
            pno = part.get('part_no', '')
            now_pnos.add(pno)

            if pno in self._seen_pnos:
                continue   # already confirmed — just track liveness below

            if pno not in self._candidates:
                # First time we see this pno — start the timer
                self._candidates[pno] = (now, part, item.get('conf', 0))
            else:
                first_t, _, best_conf = self._candidates[pno]
                best_conf = max(best_conf, item.get('conf', 0))
                self._candidates[pno] = (first_t, part, best_conf)
                if now - first_t >= self.CONFIRM_SEC:
                    # ✅ Confirmed! Graduate to history.
                    self._history.insert(0, {
                        'part':  part,
                        'conf':  best_conf,
                        'label': item.get('label', ''),
                    })
                    self._seen_pnos.add(pno)
                    del self._candidates[pno]
                    changed = True

        # Reset timers for parts that disappeared this frame
        gone = [pno for pno in self._candidates if pno not in now_pnos]
        for pno in gone:
            del self._candidates[pno]

        # Update which confirmed parts are currently live
        live_now = now_pnos & self._seen_pnos
        if live_now != self._live_pnos:
            self._live_pnos = live_now
            changed = True

        # Update status bar even without a full redraw (shows tracking count)
        if self._candidates or changed:
            self._update_status()

        if changed:
            self._redraw()

    def _reset(self):
        self._history.clear()
        self._seen_pnos.clear()
        self._live_pnos.clear()
        self._candidates.clear()
        self._redraw()

    def _update_status(self):
        """Refresh just the status bar (fast path, no card rebuild)."""
        total   = len(self._history)
        live_n  = len(self._live_pnos)
        track_n = len(self._candidates)
        if total == 0 and track_n == 0:
            self._status.config(
                text="Start camera — parts need 1-2 s to confirm", fg=MUTED)
        elif track_n:
            self._status.config(
                text=f"  {total} confirmed  ·  {live_n} live  "
                     f"·  🕒 tracking {track_n} new  ·  tap row to view",
                fg=ORANGE)
        else:
            self._status.config(
                text=f"  {total} part(s) detected  ·  {live_n} live now  "
                     f"·  tap any row to view info",
                fg=TEXT)

    def _redraw(self):
        for w in self._body.winfo_children(): w.destroy()
        self._cards.clear()

        total = len(self._history)

        if total == 0:
            self._update_status()
            tk.Label(self._body, bg=BG, fg=MUTED, font=FB,
                     text="No parts confirmed yet.\nHold a part steady for 1-2 s.",
                     pady=30, justify='center').pack()
            return

        self._update_status()

        for i, h in enumerate(self._history):
            self._make_card(i, h)

    def _make_card(self, idx, h):
        part  = h['part']
        pno   = part.get('part_no', '')
        pname = part.get('part_name', '')
        model = part.get('model', '')
        conf  = h.get('conf', 0)
        live  = pno in self._live_pnos

        # ── Card frame ─────────────────────────────────────────────────────
        CARD_BG  = '#162a4a' if live else CARD
        BADGE_BG = '#00695c' if live else '#1a3060'
        BADGE_FG = '#00e676' if live else '#90caf9'
        BADGE_TX = '📷  LIVE' if live else '📋  SEEN'

        card = tk.Frame(self._body, bg=CARD_BG,
                        highlightthickness=1,
                        highlightbackground='#00e676' if live else BORDER,
                        cursor='hand2')
        card.pack(fill='x', padx=8, pady=4)

        # ── Top row: badge + part no ────────────────────────────────────────
        top = tk.Frame(card, bg=CARD_BG, padx=10, pady=6); top.pack(fill='x')
        badge = tk.Label(top, text=BADGE_TX, bg=BADGE_BG, fg=BADGE_FG,
                         font=("Arial", 8, "bold"), padx=6, pady=2,
                         relief='flat')
        badge.pack(side='left')
        tk.Label(top, text=pno, bg=CARD_BG, fg=CYAN,
                 font=("Courier New", 10, "bold")).pack(side='left', padx=8)

        # ── Part name ───────────────────────────────────────────────────────
        mid = tk.Frame(card, bg=CARD_BG, padx=10); mid.pack(fill='x')
        tk.Label(mid, text=pname, bg=CARD_BG, fg=TEXT,
                 font=("Arial", 11, "bold"),
                 anchor='w', wraplength=360, justify='left').pack(fill='x')
        if model:
            tk.Label(mid, text=f"Model: {model}", bg=CARD_BG, fg=MUTED,
                     font=FSM, anchor='w').pack(fill='x')

        # ── Confidence bar ──────────────────────────────────────────────────
        bot = tk.Frame(card, bg=CARD_BG, padx=10, pady=6); bot.pack(fill='x')
        tk.Label(bot, text=f"Conf: {int(conf*100)}%",
                 bg=CARD_BG, fg=BADGE_FG,
                 font=("Arial", 9, "bold")).pack(side='left')
        track = tk.Frame(bot, bg=BORDER, height=6)
        track.pack(side='left', fill='x', expand=True, padx=(8,0))
        fill  = tk.Frame(track, bg='#00e676' if live else '#29b6f6', height=6)
        fill.place(relwidth=max(0.02, conf), relheight=1.0)

        # ── Hover & click ───────────────────────────────────────────────────
        def _enter(e, c=card, bg=CARD_BG):
            c.config(highlightbackground=CYAN)
            for w in c.winfo_children():
                try: w.config(bg='#1e3a5a' if live else '#1a2a3a')
                except: pass

        def _leave(e, c=card, bg=CARD_BG):
            c.config(highlightbackground='#00e676' if live else BORDER)
            for w in c.winfo_children():
                try: w.config(bg=bg)
                except: pass

        def _click(e, p=part): self.on_select(p)

        for w in [card, top, mid, bot, badge, track]:
            try:
                w.bind('<Button-1>', _click)
                w.bind('<Enter>',    _enter)
                w.bind('<Leave>',    _leave)
            except Exception: pass

        self._cards.append(card)




# ══════════════════════════════════════════════════════════════════════════════
#  ADD PART SCREEN  (3 tabs, tablet-friendly)
# ══════════════════════════════════════════════════════════════════════════════
class AddPartScreen(tk.Frame):
    def __init__(self, parent, existing=None):
        super().__init__(parent, bg=BG)
        self.existing  = existing
        self._vars     = {}
        self._cap      = None
        self._camrun   = False
        self._part_no  = existing['part_no'] if existing else None
        self._root_win = parent.winfo_toplevel()
        self._build()
        if existing: self._populate(); self._refresh_gallery()

    def _build(self):
        hdr=tk.Frame(self,bg=HDR,pady=10); hdr.pack(fill='x')
        tk.Label(hdr,text="✏️ Edit Part" if self.existing else "➕ Add New Part",
                 bg=HDR,fg='white',font=FT).pack(side='left',padx=16)
        TB(hdr,"← Back",lambda:self.master._show_db(),bg=GREY,fg='white',font=FSM,padx=12).pack(side='right',padx=14)

        sty=ttk.Style(); sty.theme_use('default')
        sty.configure('AP.TNotebook',background=BG,borderwidth=0)
        sty.configure('AP.TNotebook.Tab',background=CARD,foreground=MUTED,font=FH3,padding=[16,10])
        sty.map('AP.TNotebook.Tab',background=[('selected',HDR)],foreground=[('selected','white')])

        nb=ttk.Notebook(self,style='AP.TNotebook'); nb.pack(fill='both',expand=True,padx=6,pady=6)
        t1=tk.Frame(nb,bg=BG); nb.add(t1,text="🔧  Part Info")
        t2=tk.Frame(nb,bg=BG); nb.add(t2,text="📷  Images & Annotate")

        self._build_info(t1)
        self._build_images(t2)

    # ── Tab 1 ─────────────────────────────────────────────────────────────────
    def _build_info(self,parent):
        cv,body=make_touch_scrollable(parent,BG)
        pad=tk.Frame(body,bg=BG); pad.pack(fill='both',padx=16,pady=14)
        cols=tk.Frame(pad,bg=BG); cols.pack(fill='x')
        form=tk.Frame(cols,bg=BG); form.pack(side='left',fill='both',expand=True,padx=(0,16))

        self._sec(form,"🔧  Part Information")
        self._field(form,"Part No  *",   'part_no',    "e.g.  966626027R")
        self._field(form,"Part Name  *", 'part_name',  "e.g.  Assist Grip Handle")
        self._field(form,"Model",        'model',      "e.g.  P1324H")
        self._field(form,"Supplier",     'supplier',   "e.g.  Safety Trim - Plastique")
        self._field(form,"YOLO Class  *",'yolo_class', "e.g.  grab_handle")
        self._field(form,"Group",        'group_name', "e.g.  A")
        self._field(form,"Date",         'date',       "e.g.  06-05-2026")
        # Zone, Quantity, Reason, Judgement removed

        # Guide panel
        rp=tk.Frame(cols,bg=CARD,padx=14,pady=14,width=300); rp.pack(side='right',fill='y'); rp.pack_propagate(False)
        tk.Label(rp,text="ℹ️  YOLO Class",bg=CARD,fg=YELLOW,font=FH3).pack(anchor='w')
        tk.Frame(rp,bg=BORDER,height=1).pack(fill='x',pady=6)
        tk.Label(rp,text=("Short AI code for this part.\n\n"
                          "Rules:\n  ✅ lowercase\n  ✅ underscore\n  ❌ no spaces\n\n"
                          "Examples:\n  grab_handle\n  door_lock\n  cover_type1\n\n"
                          "After saving:\n"
                          "  📷 Tab 2 → add photos & import YOLO labels\n"
                          "  Menu → 🧠 Train AI"),
                 bg=CARD,fg=MUTED,font=FSM,justify='left').pack(anchor='w')

        bf=tk.Frame(pad,bg=BG,pady=14); bf.pack(fill='x')
        TB(bf,"💾  Save Part Info",self._save_info,bg=GREEN,fg='#000',font=FH2,padx=20).pack(side='left',padx=4)
        TB(bf,"🗑  Clear",self._clear_form,bg=GREY,fg='white',font=FB,padx=14).pack(side='left',padx=4)
        if self.existing:
            TB(bf,"❌  Delete Part",self._delete_part,bg=DRED,fg='white',font=FB,padx=14).pack(side='right',padx=4)

        # Attach virtual keyboard to all entries after build
        self.after(100, lambda: attach_kb_all(self._root_win, form))

    def _sec(self,p,title,pad=4):
        f=tk.Frame(p,bg=BG); f.pack(fill='x',pady=(pad,6))
        tk.Label(f,text=title,bg=BG,fg=CYAN,font=("Arial",11,"bold")).pack(side='left')
        tk.Frame(f,bg=BORDER,height=1).pack(side='left',fill='x',expand=True,padx=(10,0))

    def _field(self,parent,label,key,ph=''):
        row=tk.Frame(parent,bg=BG); row.pack(fill='x',pady=5)
        tk.Label(row,text=label,bg=BG,fg=MUTED,font=FSM,width=16,anchor='w').pack(side='left')
        var=tk.StringVar(); self._vars[key]=var
        e=tk.Entry(row,textvariable=var,bg=CARD,fg=MUTED,insertbackground=TEXT,
                   font=FB,relief='flat',bd=0,highlightthickness=1,
                   highlightbackground=BORDER,highlightcolor=CYAN)
        e.pack(side='left',fill='x',expand=True,ipady=8,padx=8)
        if ph:
            e.insert(0,ph)
            e.bind('<FocusIn>',lambda ev,_e=e,_p=ph:(_e.delete(0,'end'),_e.config(fg=TEXT)) if _e.get()==_p else None)
            e.bind('<FocusOut>',lambda ev,_e=e,_p=ph:(_e.insert(0,_p),_e.config(fg=MUTED)) if not _e.get() else None)
        attach_kb(self._root_win, e)

    def _gv(self, key):
        ph = {
            'part_no':    'e.g.  966626027R',
            'part_name':  'e.g.  Assist Grip Handle',
            'model':      'e.g.  P1324H',
            'supplier':   'e.g.  Safety Trim - Plastique',
            'yolo_class': 'e.g.  grab_handle',
            'group_name': 'e.g.  A',
            'date':       'e.g.  06-05-2026',
        }
        if key not in self._vars:
            return ''   # field was removed — return empty safely
        v = self._vars[key].get().strip()
        return '' if v == ph.get(key, '') else v

    def _get_yolo_class(self):
        if self._part_no:
            p=get_part_by_no(self._part_no)
            if p: return p.get('yolo_class','') or ''
        return self._gv('yolo_class')

    def _populate(self):
        p=self.existing
        for k in ('part_no','part_name','model','supplier','yolo_class','group_name','date'):
            if k in self._vars: self._vars[k].set(p.get(k,'') or '')
        # quantity/zone/reason/judgement fields removed

    def _save_info(self):
        pno  = self._gv('part_no')
        pname= self._gv('part_name')
        yolo = self._gv('yolo_class')
        if not pno:
            messagebox.showerror("Required","Part No is required.");   return
        if not pname:
            messagebox.showerror("Required","Part Name is required."); return
        if not yolo:
            messagebox.showerror("Required","YOLO Class is required."); return
        data = dict(
            part_no    = pno,
            part_name  = pname,
            model      = self._gv('model'),
            supplier   = self._gv('supplier'),
            yolo_class = yolo,
            group_name = self._gv('group_name'),
            date       = self._gv('date'),
            zone='', quantity=0, judgement='', reason='', image_path=''
        )
        add_part(data)
        self._part_no = pno
        if not self.existing:
            self.existing = {'part_no': pno}
        messagebox.showinfo("Saved",
            f"✅ Part '{pno}' saved!\n\n"
            "Next:\n  📷 Images & Annotate tab\n"
            "  → Upload images, annotate, or import YOLO labels")

    def _clear_form(self):
        for var in self._vars.values(): var.set('')

    def _delete_part(self):
        pno=self.existing.get('part_no','')
        if messagebox.askyesno("Delete",f"Delete '{pno}' and all images?"):
            delete_part(pno); messagebox.showinfo("Deleted",f"'{pno}' deleted.")
            self.master._show_db()

    # ── Tab 2: Images & Annotate (includes YOLO upload + Export) ─────────────
    def _build_images(self, parent):
        # ── Top action bar ────────────────────────────────────────────────────
        tb = tk.Frame(parent, bg=CARD, padx=12, pady=10)
        tb.pack(fill='x')
        tk.Label(tb, text="📷  Images & Annotations",
                 bg=CARD, fg=CYAN, font=FH3).pack(side='left')

        # Right-side action buttons (all in one row)
        rf = tk.Frame(tb, bg=CARD)
        rf.pack(side='right')
        TB(rf, "📁 Upload\nImages",   self._upload_imgs,
           bg=BLU,  fg='white', font=FSM, padx=10).pack(side='left', padx=2)
        TB(rf, "📸 Capture\nCamera",  self._open_cam,
           bg=GREY, fg='white', font=FSM, padx=10).pack(side='left', padx=2)
        TB(rf, "📄 Upload\nYOLO .txt",self._upload_yolo_pairs,
           bg='#5c35a8', fg='white', font=FSM, padx=10).pack(side='left', padx=2)
        TB(rf, "📦 Upload\nZIP",      self._upload_yolo_zip,
           bg=TEAL,  fg='white', font=FSM, padx=10).pack(side='left', padx=2)
        TB(rf, "📤 Export\nAll",      self._export_all,
           bg='#e65100', fg='white', font=FSM, padx=10).pack(side='left', padx=2)
        TB(rf, "🧠 Train\nAI Model",  self._open_train,
           bg=GREEN, fg='#000', font=FSM, padx=10).pack(side='left', padx=2)

        # ── YOLO format hint strip ────────────────────────────────────────────
        hint = tk.Frame(parent, bg='#0a1f38', padx=12, pady=6)
        hint.pack(fill='x')
        tk.Label(hint,
                 text="📄 Upload YOLO .txt — select image + matching .txt files together  |  "
                      "📦 Upload ZIP — zip containing images/ and labels/ folders",
                 bg='#0a1f38', fg=MUTED, font=("Arial", 8)).pack(side='left')

        # ── Selection bar (stats + select-all + delete selected) ────────────
        sel_bar = tk.Frame(parent, bg=PANEL, padx=10, pady=6)
        sel_bar.pack(fill='x')
        self.stats_lbl = tk.Label(sel_bar, text="", bg=PANEL, fg=MUTED,
                                  font=FSM, anchor='w')
        self.stats_lbl.pack(side='left')
        # Selection controls
        self._sel_vars = {}   # img_id -> BooleanVar
        TB(sel_bar, "☑ Select All",   self._select_all,
           bg=GREY, fg='white', font=FSM, padx=10).pack(side='right', padx=4)
        TB(sel_bar, "☐ Deselect All", self._deselect_all,
           bg=GREY, fg='white', font=FSM, padx=10).pack(side='right', padx=4)
        self.del_sel_btn = TB(sel_bar, "🗑 Delete Selected",
                               self._delete_selected,
                               bg=DRED, fg='white', font=FSM, padx=10)
        self.del_sel_btn.pack(side='right', padx=4)

        # ── Scrollable image gallery ──────────────────────────────────────────
        outer = tk.Frame(parent, bg=BG)
        outer.pack(fill='both', expand=True)
        self._gc = tk.Canvas(outer, bg=BG, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient='vertical', command=self._gc.yview)
        self._gc.configure(yscrollcommand=vsb.set)
        vsb.pack(side='right', fill='y')
        self._gc.pack(fill='both', expand=True)
        self._gf = tk.Frame(self._gc, bg=BG)
        self._gwid = self._gc.create_window((0, 0), window=self._gf, anchor='nw')
        self._gc.bind('<Configure>',
                      lambda e: (self._gc.itemconfig(self._gwid, width=e.width),
                                 self._refresh_gallery()))
        self._gf.bind('<Configure>',
                      lambda e: self._gc.configure(
                          scrollregion=self._gc.bbox('all')))
        TouchScroll(self._gc)
        register_scroll_canvas(self._gc)   # global mousewheel scrolls gallery

        # ── Import log (collapsible, shown after import) ──────────────────────
        self._import_log_frame = tk.Frame(parent, bg=PANEL)
        # (packed only when needed)
        self.import_log = tk.Text(self._import_log_frame, bg='#050e1a',
                                  fg=GREEN, font=FM, height=5,
                                  relief='flat', state='disabled')
        self.import_log.pack(fill='x', padx=8, pady=4)
        TB(self._import_log_frame, "✖ Hide Log",
           self._hide_log, bg=GREY, fg='white', font=FSM, padx=8).pack(anchor='e', padx=8, pady=2)

    def _refresh_gallery(self):
        for w in self._gf.winfo_children(): w.destroy()
        if not self._part_no:
            tk.Label(self._gf, text="⚠️  Save Part Info first.",
                     bg=BG, fg=YELLOW, font=FH2, pady=40).pack()
            return
        images   = get_part_images(self._part_no)
        ann      = [i for i in images if i['annotated']]
        self.stats_lbl.config(
            text=f"  {len(images)} images   |   {len(ann)} annotated"
                 f"   |   {len(images)-len(ann)} remaining")
        if not images:
            tk.Label(self._gf,
                     text="No images yet.\nUpload or capture photos.",
                     bg=BG, fg=MUTED, font=FH2, pady=40).pack()
            return

        self._sel_vars = {}   # reset selection state
        yolo_cls = self._get_yolo_class()
        COLS = 3
        for idx, img_rec in enumerate(images):
            r, c = divmod(idx, COLS)
            card = tk.Frame(self._gf, bg=CARD, padx=8, pady=8)
            card.grid(row=r, column=c, padx=8, pady=8, sticky='nsew')
            self._gf.grid_columnconfigure(c, weight=1)

            # Thumbnail — full image, no fixed height so PIL image fills naturally
            th = tk.Label(card, bg=PANEL, fg=MUTED, text="…", font=FSM)
            th.pack(fill='x')
            self._load_th(th, img_rec['image_path'])

            # Source badge + selection checkbox in same row
            src   = img_rec.get('source', '')
            badge = ('📂 YOLO Import' if src == 'yolo_import'
                     else '📸 Camera' if src == 'camera'
                     else '📁 Upload')
            badge_row = tk.Frame(card, bg=CARD)
            badge_row.pack(fill='x')
            tk.Label(badge_row, text=badge, bg=CARD, fg=MUTED,
                     font=("Arial", 8)).pack(side='left')
            # Per-image selection checkbox
            sel_var = tk.BooleanVar(value=False)
            self._sel_vars[img_rec['id']] = sel_var
            tk.Checkbutton(badge_row, variable=sel_var,
                           bg=CARD, fg=TEXT, selectcolor=PANEL,
                           activebackground=CARD,
                           text="Select", font=("Arial",8)
                           ).pack(side='right')

            # Annotated status
            if img_rec['annotated']:
                tk.Label(card, text="✅ Annotated",
                         bg='#1b4a2a', fg=GREEN,
                         font=("Arial", 9, "bold"), pady=2).pack(fill='x')
            else:
                tk.Label(card, text="⚠️ Not annotated",
                         bg='#3a2010', fg=ORANGE,
                         font=("Arial", 9, "bold"), pady=2).pack(fill='x')

            # Filename
            fname = os.path.basename(img_rec['image_path'])
            tk.Label(card, text=fname[:24], bg=CARD, fg=MUTED,
                     font=("Arial", 8), wraplength=160).pack()

            # Action buttons row 1: Annotate + Delete
            bf1 = tk.Frame(card, bg=CARD)
            bf1.pack(fill='x', pady=(6, 2))
            TB(bf1, "✏️ Annotate",
               lambda r=img_rec, cl=yolo_cls: self._open_ann(r, cl),
               bg=BLU, fg='white', font=FSM, padx=6
               ).pack(side='left', expand=True, fill='x', padx=1)
            TB(bf1, "🗑",
               lambda r=img_rec: self._del_img(r),
               bg=DRED, fg='white', font=FSM, padx=8
               ).pack(side='right', padx=1)

            # Action buttons row 2: Export this image+label
            bf2 = tk.Frame(card, bg=CARD)
            bf2.pack(fill='x', pady=(0, 0))
            TB(bf2, "📤 Export Image + Label",
               lambda r=img_rec: self._export_one(r),
               bg='#e65100', fg='white', font=("Arial", 8),
               padx=6).pack(fill='x', padx=1)

    def _select_all(self):
        for var in self._sel_vars.values():
            var.set(True)

    def _deselect_all(self):
        for var in self._sel_vars.values():
            var.set(False)

    def _delete_selected(self):
        selected = [img_id for img_id, var in self._sel_vars.items() if var.get()]
        if not selected:
            messagebox.showinfo("Nothing Selected",
                "Tap the Select checkbox on images you want to delete,\n"
                "or tap Select All then Delete Selected.")
            return
        if not messagebox.askyesno("Delete Selected",
                f"Delete {len(selected)} selected image(s)?\nThis cannot be undone."):
            return
        from database.db_manager import get_part_images as _gpi
        # Build id→record map for path lookup
        all_imgs = _gpi(self._part_no) if self._part_no else []
        img_map  = {i['id']: i for i in all_imgs}
        for img_id in selected:
            rec = img_map.get(img_id)
            if rec:
                delete_part_image(img_id)
                # Optionally remove the file from disk
                try:
                    if rec['image_path'] and os.path.exists(rec['image_path']):
                        os.remove(rec['image_path'])
                except Exception:
                    pass
        self._refresh_gallery()

    def _load_th(self,lbl,path):
        if not path or not os.path.exists(path): return
        try:
            img=Image.open(path).convert('RGB')
            # Compute width of the card column (~1/3 of gallery minus padding)
            gw = self._gc.winfo_width() or 600
            col_w = max(160, (gw // 3) - 32)
            # Scale image proportionally to fill the column width fully
            ow, oh = img.size
            scale = col_w / ow
            nh = max(1, int(oh * scale))
            img = img.resize((col_w, nh), Image.LANCZOS)
            ph=ImageTk.PhotoImage(img); lbl.config(image=ph,text=''); lbl.image=ph
        except Exception: pass

    def _upload_imgs(self):
        if not self._part_no: messagebox.showwarning("Save First","Save Part Info first."); return
        paths=filedialog.askopenfilenames(title="Select Part Images",
            filetypes=[("Images","*.jpg *.jpeg *.png *.bmp *.webp"),("All","*.*")])
        if not paths: return
        for path in paths:
            ext=os.path.splitext(path)[1]; ts=datetime.now().strftime('%Y%m%d%H%M%S%f')
            dest=os.path.join(PARTS_IMG_DIR,f"{self._part_no}_{ts}{ext}")
            shutil.copy2(path,dest); add_part_image(self._part_no,dest,source='upload'); time.sleep(0.001)
        try:
            from similarity_engine import invalidate; invalidate(self._part_no)
        except Exception: pass
        self._refresh_gallery()

    def _open_cam(self):
        if not self._part_no: messagebox.showwarning("Save First","Save Part Info first."); return
        CamCaptureWindow(self,self._part_no,self._get_yolo_class(),on_done=self._refresh_gallery)

    def _open_ann(self, img_rec, yolo_cls):
        if not yolo_cls:
            messagebox.showwarning("YOLO Class", "Set YOLO Class first."); return
        # Build the full image list for this part so the annotator can auto-advance
        images    = get_part_images(self._part_no) if self._part_no else [img_rec]
        img_index = next((i for i, r in enumerate(images)
                          if r['id'] == img_rec['id']), 0)
        AnnotatorWindow(self, img_rec, yolo_cls,
                        on_save_cb=self._refresh_gallery,
                        img_list=images,
                        img_index=img_index)

    def _del_img(self, img_rec):
        if messagebox.askyesno("Delete", "Delete this image?"):
            delete_part_image(img_rec['id'])
            self._refresh_gallery()

    # ── YOLO pair upload (Mode A) ──────────────────────────────────────────────
    def _upload_yolo_pairs(self):
        if not self._part_no:
            messagebox.showwarning("Save First", "Save Part Info first."); return
        paths = filedialog.askopenfilenames(
            title="Select image files AND their .txt files together",
            filetypes=[("Images & Labels", "*.jpg *.jpeg *.png *.bmp *.txt"),
                       ("All files", "*.*")])
        if not paths: return
        imgs = {os.path.splitext(p)[0]: p for p in paths if not p.endswith('.txt')}
        txts = {os.path.splitext(p)[0]: p for p in paths if p.endswith('.txt')}
        ok = skip = 0
        self._show_log()
        self._log_msg(f"\n── Importing YOLO pairs for [{self._part_no}] ──")
        for stem, img_p in imgs.items():
            txt_p = txts.get(stem)
            if not txt_p:
                self._log_msg(f"  ⚠  No .txt for {os.path.basename(img_p)} — skipped")
                skip += 1; continue
            if self._import_pair(img_p, txt_p, self._part_no):
                self._log_msg(f"  ✅  {os.path.basename(img_p)}")
                ok += 1
            else:
                skip += 1
        self._log_msg(f"Done: ✅ {ok} imported   ⚠ {skip} skipped")
        from similarity_engine import invalidate; invalidate(self._part_no)
        self._refresh_gallery()
        messagebox.showinfo("Import Complete", f"Imported {ok} image(s).\nSkipped {skip}.")

    # ── YOLO ZIP upload (Mode B) ──────────────────────────────────────────────
    def _upload_yolo_zip(self):
        if not self._part_no:
            messagebox.showwarning("Save First", "Save Part Info first."); return
        zpath = filedialog.askopenfilename(
            title="Select ZIP dataset",
            filetypes=[("ZIP", "*.zip"), ("All", "*.*")])
        if not zpath: return
        tmp = os.path.join(ROOT, '_zip_tmp')
        shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
        self._show_log()
        self._log_msg(f"\n── Unzipping {os.path.basename(zpath)} ──")
        try:
            with zipfile.ZipFile(zpath) as z: z.extractall(tmp)
        except Exception as e:
            self._log_msg(f"  ❌ ZIP error: {e}"); return
        img_map = {}; lbl_map = {}
        for root, _, files in os.walk(tmp):
            for fn in files:
                fp = os.path.join(root, fn)
                stem = os.path.splitext(fn)[0]
                ext  = os.path.splitext(fn)[1].lower()
                if ext in {'.jpg','.jpeg','.png','.bmp'}: img_map[stem] = fp
                elif ext == '.txt': lbl_map[stem] = fp
        ok = skip = 0
        for stem, img_p in img_map.items():
            lbl_p = lbl_map.get(stem)
            if not lbl_p:
                self._log_msg(f"  ⚠  No .txt for {os.path.basename(img_p)}")
                skip += 1; continue
            if self._import_pair(img_p, lbl_p, self._part_no):
                self._log_msg(f"  ✅  {os.path.basename(img_p)}"); ok += 1
            else:
                skip += 1
        shutil.rmtree(tmp, ignore_errors=True)
        self._log_msg(f"Done: ✅ {ok} imported   ⚠ {skip} skipped")
        from similarity_engine import invalidate; invalidate(self._part_no)
        self._refresh_gallery()
        messagebox.showinfo("Import Complete", f"Imported {ok} image(s) from ZIP.\nSkipped {skip}.")

    def _import_pair(self, img_path, txt_path, pno):
        """Validate and import one image+YOLO-label pair."""
        if not os.path.exists(img_path): return False
        valid = []
        try:
            for raw in open(txt_path):
                pts = raw.strip().split()
                if len(pts) != 5: continue
                try:
                    vals = list(map(float, pts))
                    if all(0.0 <= v <= 1.0 for v in vals[1:]):
                        pts[0] = '0'   # remap class_id to 0
                        valid.append(' '.join(pts))
                except ValueError: continue
        except Exception: return False
        if not valid: return False
        ts      = datetime.now().strftime('%Y%m%d%H%M%S%f')
        ext     = os.path.splitext(img_path)[1].lower() or '.jpg'
        dst_img = os.path.join(PARTS_IMG_DIR, f"{pno}_{ts}{ext}")
        shutil.copy2(img_path, dst_img)
        ldir    = os.path.join(ROOT, 'dataset', 'labels_raw')
        os.makedirs(ldir, exist_ok=True)
        dst_lbl = os.path.join(ldir, f"{pno}_{ts}.txt")
        open(dst_lbl, 'w').write('\n'.join(valid))
        add_part_image(pno, dst_img, dst_lbl, annotated=1, source='yolo_import')
        return True

    # ── Export one image + label ──────────────────────────────────────────────
    def _export_one(self, img_rec):
        """Export one image and its YOLO .txt label to a folder."""
        dest_dir = filedialog.askdirectory(title="Select folder to export to")
        if not dest_dir: return
        img_path = img_rec['image_path']
        lbl_path = img_rec.get('label_path', '')
        fname    = os.path.basename(img_path)
        shutil.copy2(img_path, os.path.join(dest_dir, fname))
        if lbl_path and os.path.exists(lbl_path):
            lbl_name = os.path.splitext(fname)[0] + '.txt'
            shutil.copy2(lbl_path, os.path.join(dest_dir, lbl_name))
            messagebox.showinfo("Exported",
                f"✅ Exported:\n  {fname}\n  {lbl_name}\n\nTo: {dest_dir}")
        else:
            messagebox.showinfo("Exported",
                f"✅ Exported image (no annotation yet):\n  {fname}\n\nTo: {dest_dir}")

    # ── Export ALL images + labels for this part ──────────────────────────────
    def _export_all(self):
        if not self._part_no:
            messagebox.showwarning("Save First", "Save Part Info first."); return
        images = get_part_images(self._part_no)
        if not images:
            messagebox.showinfo("No Images", "No images to export."); return
        dest_dir = filedialog.askdirectory(title="Select folder to export all images+labels")
        if not dest_dir: return
        exported_imgs = exported_lbls = 0
        for img_rec in images:
            img_path = img_rec['image_path']
            lbl_path = img_rec.get('label_path', '')
            if os.path.exists(img_path):
                fname = os.path.basename(img_path)
                shutil.copy2(img_path, os.path.join(dest_dir, fname))
                exported_imgs += 1
                if lbl_path and os.path.exists(lbl_path):
                    lbl_name = os.path.splitext(fname)[0] + '.txt'
                    shutil.copy2(lbl_path, os.path.join(dest_dir, lbl_name))
                    exported_lbls += 1
        messagebox.showinfo("Export Complete",
            f"✅ Exported {exported_imgs} images\n"
            f"   {exported_lbls} YOLO .txt labels\n\n"
            f"To folder:\n{dest_dir}")

    # ── Import log helpers ────────────────────────────────────────────────────
    def _show_log(self):
        self._import_log_frame.pack(fill='x', before=self._gc.master
                                    if hasattr(self, '_gc') else None)
        self.import_log.config(state='normal')
        self.import_log.delete(1.0, 'end')
        self.import_log.config(state='disabled')

    def _hide_log(self):
        self._import_log_frame.pack_forget()

    def _log_msg(self, msg):
        self.import_log.config(state='normal')
        self.import_log.insert('end', msg + '\n')
        self.import_log.see('end')
        self.import_log.config(state='disabled')

    def _open_train(self): TrainingWindow(self.master)
    def on_hide(self): pass


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA CAPTURE
# ══════════════════════════════════════════════════════════════════════════════
class CamCaptureWindow(tk.Toplevel):
    def __init__(self,parent,part_no,yolo_class,on_done=None):
        super().__init__(parent); self.title("📸  Camera Capture")
        self.configure(bg=BG); self.geometry("700x540")
        self.part_no=part_no; self.yolo_class=yolo_class; self.on_done=on_done
        self._cap=None; self._running=False; self._count=0
        self._build(); self.grab_set()

    def _build(self):
        hdr=tk.Frame(self,bg=HDR,pady=10); hdr.pack(fill='x')
        tk.Label(hdr,text="📸  Capture Training Images",bg=HDR,fg='white',font=FH2).pack(side='left',padx=16)
        self.cam=tk.Label(self,bg=PANEL,fg=MUTED,text="Camera Off",font=FB); self.cam.pack(fill='both',expand=True,padx=10,pady=8)
        ctrl=tk.Frame(self,bg=BG,pady=8); ctrl.pack(fill='x',padx=10)
        self.bs=TB(ctrl,"▶ Open",self._start,bg=BLU,fg='white',padx=16); self.bs.pack(side='left',padx=4)
        self.bc=TB(ctrl,"📸 Capture",self._snap,bg=GREEN,fg='#000',font=FH2,padx=16,state='disabled'); self.bc.pack(side='left',padx=4)
        self.bx=TB(ctrl,"⏹ Stop",self._stop,bg=DRED,fg='white',padx=16,state='disabled'); self.bx.pack(side='left',padx=4)
        self.cl=tk.Label(ctrl,text="Captured: 0",bg=BG,fg=CYAN,font=FH3); self.cl.pack(side='right',padx=12)
        tk.Label(self,text="Capture 30+ photos — front, side, top, close-up, different lighting.",
                 bg=CARD,fg=MUTED,font=FSM,pady=6,padx=10,anchor='w').pack(fill='x')
        TB(self,"✅  Done",self._done,bg=CYAN,fg='#000',font=FH2,padx=20).pack(pady=10)

    def _start(self):
        self._cap=cv2.VideoCapture(0)
        if not self._cap.isOpened(): messagebox.showerror("Camera","Cannot open camera."); return
        self._running=True; self.bs.config(state='disabled')
        self.bc.config(state='normal'); self.bx.config(state='normal')
        threading.Thread(target=self._loop,daemon=True).start()

    def _loop(self):
        while self._running and self._cap:
            ret,frame=self._cap.read()
            if not ret: break
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            img=Image.fromarray(rgb); img.thumbnail((660,420),Image.BILINEAR)
            ph=ImageTk.PhotoImage(img); self.cam.config(image=ph,text=''); self.cam.image=ph

    def _snap(self):
        if not self._cap: return
        ret,frame=self._cap.read()
        if ret:
            ts=datetime.now().strftime('%Y%m%d%H%M%S%f')
            dest=os.path.join(PARTS_IMG_DIR,f"{self.part_no}_{ts}.jpg")
            cv2.imwrite(dest,frame); add_part_image(self.part_no,dest,source='camera')
            self._count+=1; self.cl.config(text=f"Captured: {self._count}")

    def _stop(self):
        self._running=False
        if self._cap: self._cap.release(); self._cap=None
        self.bs.config(state='normal'); self.bc.config(state='disabled'); self.bx.config(state='disabled')
        self.cam.config(image='',text='Camera Off'); self.cam.image=None

    def _done(self):
        self._stop()
        try:
            from similarity_engine import invalidate; invalidate(self.part_no)
        except Exception: pass
        if self.on_done: self.on_done()
        self.destroy()


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING WINDOW
# ══════════════════════════════════════════════════════════════════════════════
class TrainingWindow(tk.Toplevel):
    def __init__(self,parent):
        super().__init__(parent); self.title("🧠  Train AI Model")
        self.configure(bg=BG)
        self._training=False
        # Start maximised so footer (Start Training) is always visible
        try:
            self.state('zoomed')
        except Exception:
            try:
                self.attributes('-zoomed', True)
            except Exception:
                self.geometry("1200x800")
        self._build(); self.grab_set()

    def _build(self):
        # ── Header (always visible top) ──────────────────────────────────────
        hdr=tk.Frame(self,bg=HDR,pady=10); hdr.pack(side='top',fill='x')
        tk.Label(hdr,text="🧠  Train AI Model",bg=HDR,fg='white',font=FT).pack(side='left',padx=16)

        # ── Footer (packed BEFORE scrollable body → always visible at bottom) ─
        footer=tk.Frame(self,bg='#0a1628',pady=12); footer.pack(side='bottom',fill='x')
        self.btn=TB(footer,"🚀  Start Training",self._start,bg=GREEN,fg='#000',
                    font=("Arial",14,"bold"),padx=28)
        self.btn.pack(side='left',padx=12)
        TB(footer,"✖  Close",self.destroy,bg=DRED,fg='white',font=FH2,padx=16).pack(side='left',padx=4)
        self.rlbl=tk.Label(footer,text="",bg='#0a1628',fg=GREEN,font=FH3)
        self.rlbl.pack(side='right',padx=16)

        # ── Scrollable body fills remaining space ─────────────────────────────
        _,body=make_touch_scrollable(self,BG)
        pad=tk.Frame(body,bg=BG); pad.pack(fill='x',padx=16,pady=10)

        self.sum_lbl=tk.Label(pad,text="",bg=CARD,fg=TEXT,font=FB,justify='left',padx=14,pady=10,anchor='w')
        self.sum_lbl.pack(fill='x',pady=(0,8)); self._refresh_summary()

        sf=tk.Frame(pad,bg=CARD,padx=16,pady=14); sf.pack(fill='x',pady=(0,8))
        tk.Label(sf,text="⚙️  Training Settings",bg=CARD,fg=CYAN,font=FH3).pack(anchor='w',pady=(0,10))

        r0=tk.Frame(sf,bg=CARD); r0.pack(fill='x',pady=4)
        tk.Label(r0,text="Model:",bg=CARD,fg=MUTED,font=FB,width=12,anchor='w').pack(side='left')
        self.msz=tk.StringVar(value='s')
        for v,l in [('n','Nano (fast, ~10min)'),('s','Small ★ (~25min)'),('m','Medium (best, ~50min)')]:
            tk.Radiobutton(r0,text=l,variable=self.msz,value=v,bg=CARD,fg=TEXT,selectcolor=PANEL,
                           font=FB,activebackground=CARD).pack(side='left',padx=10)

        r1=tk.Frame(sf,bg=CARD); r1.pack(fill='x',pady=4)
        tk.Label(r1,text="Epochs:",bg=CARD,fg=MUTED,font=FB,width=12,anchor='w').pack(side='left')
        self.ep=tk.IntVar(value=80)
        tk.Spinbox(r1,from_=10,to=300,textvariable=self.ep,width=6,bg=PANEL,fg=TEXT,font=FB,
                   relief='flat',buttonbackground=PANEL).pack(side='left',padx=8)
        tk.Label(r1,text="80=good   150=better   200+=best",bg=CARD,fg=MUTED,font=FSM).pack(side='left',padx=8)

        r2=tk.Frame(sf,bg=CARD); r2.pack(fill='x',pady=4)
        tk.Label(r2,text="Img Size:",bg=CARD,fg=MUTED,font=FB,width=12,anchor='w').pack(side='left')
        self.isz=tk.IntVar(value=640)
        ttk.Combobox(r2,textvariable=self.isz,values=[320,416,512,640],width=7,state='readonly',font=FB).pack(side='left',padx=8)

        # Progress
        pf=tk.Frame(pad,bg=CARD,padx=16,pady=12); pf.pack(fill='x',pady=(0,6))
        self.plbl=tk.Label(pf,text="Ready to train",bg=CARD,fg=TEXT,font=FB); self.plbl.pack(anchor='w')
        sty=ttk.Style(); sty.configure("TrPB.Horizontal.TProgressbar",troughcolor=PANEL,background=GREEN,thickness=20)
        self.pb=ttk.Progressbar(pf,mode='determinate',style='TrPB.Horizontal.TProgressbar'); self.pb.pack(fill='x',pady=6)
        self.pct=tk.Label(pf,text="0%",bg=CARD,fg=GREEN,font=FH3); self.pct.pack(anchor='e')

        # Log
        lhdr=tk.Frame(pad,bg=BG); lhdr.pack(fill='x',pady=(4,2))
        tk.Label(lhdr,text="📋  Training Log",bg=BG,fg=MUTED,font=FSM).pack(side='left')
        TB(lhdr,"🗑 Clear",lambda:(self.log.config(state='normal'),self.log.delete(1.0,'end'),self.log.config(state='disabled')),
           bg=GREY,fg='white',font=FSM,padx=10).pack(side='right')
        lf=tk.Frame(pad,bg=BG); lf.pack(fill='both',expand=True)
        self.log=tk.Text(lf,bg='#050e1a',fg=GREEN,font=FM,relief='flat',state='disabled',wrap='word',height=12)
        lsb=ttk.Scrollbar(lf,orient='vertical',command=self.log.yview); self.log.configure(yscrollcommand=lsb.set)
        lsb.pack(side='right',fill='y'); self.log.pack(fill='both',expand=True)

    def _refresh_summary(self):
        imgs=get_all_annotated_images(); cls=get_all_yolo_classes()
        if not imgs: self.sum_lbl.config(text="⚠️  No annotated images. Add images and annotate first.",fg=RED,bg=CARD); return
        by_cls={}
        for i in imgs: by_cls[i['yolo_class']]=by_cls.get(i['yolo_class'],0)+1
        lines=["📊  Dataset:"]
        for c,n in by_cls.items():
            st="✅" if n>=30 else ("⚠️" if n>=10 else "❌")
            hint="excellent" if n>=50 else ("good" if n>=30 else ("add more" if n>=10 else "too few — need 30+"))
            lines.append(f"  {st}  {c}:  {n} images  ({hint})")
        lines.append(f"\nTotal: {len(imgs)}   Classes: {len(cls)}")
        self.sum_lbl.config(text='\n'.join(lines),fg=TEXT,bg=CARD)

    def _add_log(self,msg):
        def _d(): self.log.config(state='normal'); self.log.insert('end',msg+'\n'); self.log.see('end'); self.log.config(state='disabled')
        self.after(0,_d)

    def _set_prog(self,pct,msg):
        def _d(): self.pb['value']=pct; self.plbl.config(text=msg); self.pct.config(text=f"{pct}%")
        self.after(0,_d)

    def _start(self):
        if self._training: return
        self._training=True; self.btn.config(state='disabled',text="⏳  Training...")
        self.rlbl.config(text=""); self.log.config(state='normal'); self.log.delete(1.0,'end'); self.log.config(state='disabled')
        threading.Thread(target=self._thread,args=(self.ep.get(),self.isz.get(),self.msz.get()),daemon=True).start()

    def _thread(self,ep,isz,msz):
        from training_pipeline import run_training
        r=run_training(epochs=ep,img_size=isz,model_size=msz,progress_cb=self._set_prog,log_cb=self._add_log)
        def _done():
            self._training=False; self.btn.config(state='normal',text="🚀  Start Training")
            if r['success']:
                g=r.get('grade','?'); col=GREEN if g in('A','B') else ORANGE if g=='C' else RED
                self.rlbl.config(text=f"✅  Grade:{g}  mAP50:{r.get('mAP50',0):.3f}",fg=col)
                try:
                    m=self.master._screens.get('main')
                    if m and hasattr(m,'detector'): m.detector.reload(); m.sys_lbl.config(text=f"● Model reloaded",fg=GREEN)
                except Exception: pass
                messagebox.showinfo("Done",r['message'],parent=self)
            else: self.rlbl.config(text="❌ Failed",fg=RED); messagebox.showerror("Failed",r['message'],parent=self)
        self.after(0,_done)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN DETECTION SCREEN
# ══════════════════════════════════════════════════════════════════════════════
class MainDetectionScreen(tk.Frame):
    def __init__(self,parent):
        super().__init__(parent,bg=BG)
        self.detector=PartDetector(); self.cap=None; self.running=False
        self._overlay=[]; self._selected=None; self._last_logged={}
        self._build()

    def _build(self):
        body=tk.Frame(self,bg=BG); body.pack(fill='both',expand=True,padx=8,pady=6)
        left=tk.Frame(body,bg=CARD); left.pack(side='left',fill='both',expand=True,padx=(0,6))
        tk.Frame(left,bg=HDR,pady=8).pack(fill='x')
        tk.Label(left.winfo_children()[-1],text="LIVE CAMERA FEED",bg=HDR,fg='white',font=FH2).pack()
        self.canvas=tk.Canvas(left,bg='#000',highlightthickness=0,cursor='hand2')
        self.canvas.pack(fill='both',expand=True)
        self.canvas.bind('<Button-1>',self._on_click)
        sb=tk.Frame(left,bg='#1a2a3a',pady=4); sb.pack(fill='x')
        self.det_lbl=tk.Label(sb,text="Objects Detected : 0",bg='#1a2a3a',fg=TEXT,font=FB); self.det_lbl.pack(side='left',padx=16)
        self.fps_lbl=tk.Label(sb,text="FPS : 0.0",bg='#1a2a3a',fg=TEXT,font=FB); self.fps_lbl.pack(side='right',padx=16)

        right=tk.Frame(body,bg=BG,width=430); right.pack(side='right',fill='y'); right.pack_propagate(False)
        info_top=tk.Frame(right,bg=CARD); info_top.pack(fill='x')
        tk.Frame(info_top,bg=HDR,pady=7).pack(fill='x')
        tk.Label(info_top.winfo_children()[-1],text="PART INFORMATION",bg=HDR,fg='white',font=FH2).pack()
        self.part_img=tk.Label(info_top,bg='#1a2a3a',fg=MUTED,text="Tap a detected part",font=FSM)
        self.part_img.pack(fill='x',padx=8,pady=4)
        self._dvars={}
        df=tk.Frame(info_top,bg=CARD,padx=10); df.pack(fill='x')
        for lbl,key,col in [("Part No",'part_no',CYAN),("Part Name",'part_name',ORANGE),("Model",'model',CYAN)]:
            row=tk.Frame(df,bg=CARD); row.pack(fill='x',pady=3)
            tk.Label(row,text=lbl,bg=CARD,fg=TEXT,font=FSM,width=11,anchor='w').pack(side='left')
            tk.Label(row,text=":",bg=CARD,fg=MUTED,font=FSM).pack(side='left',padx=2)
            var=tk.StringVar(value="—"); lv=tk.Label(row,textvariable=var,bg=CARD,fg=col,font=("Arial",10,"bold"),anchor='w')
            lv.pack(side='left',fill='x',expand=True); self._dvars[key]=(var,lv,col)
        TB(info_top,"📋 Export Details",self._export,bg=GREEN,fg='#000',font=FSM).pack(fill='x',padx=8,pady=8)

        self._sim=SimilarityPanel(right,on_select_cb=self._on_sim_click)
        self._sim.pack(fill='both',expand=True,pady=(4,0))

        bb=tk.Frame(self,bg=PANEL,pady=8); bb.pack(fill='x',side='bottom')
        self.btn_start=TB(bb,"▶  Start Camera",self.start_camera,bg=BLU,fg='white',padx=18); self.btn_start.pack(side='left',padx=6)
        self.btn_stop=TB(bb,"⏹  Stop",self.stop_camera,bg=DRED,fg='white',padx=18,state='disabled'); self.btn_stop.pack(side='left',padx=6)
        TB(bb,"📸  Capture",self.take_screenshot,bg=GREY,fg='white',padx=14).pack(side='left',padx=6)
        TB(bb,"🧠  Train",lambda:TrainingWindow(self.master),bg=GREEN,fg='#000',padx=14).pack(side='left',padx=6)
        TB(bb,"🚪  Exit",self.master.on_close,bg=GREY,fg='white',padx=14).pack(side='left',padx=6)
        self.sys_lbl=tk.Label(bb,text="● Status :  Ready",bg=PANEL,fg=GREEN,font=FB); self.sys_lbl.pack(side='left',padx=14)
        tk.Label(bb,text="Version 1.0",bg=PANEL,fg=MUTED,font=FSM).pack(side='right',padx=14)

    def start_camera(self):
        if self.running: return
        if not self.detector.is_ready():
            if not messagebox.askyesno("No Trained Model",
                "⚠️  No custom model found.\n\n"
                "Camera will open but detection won't be accurate.\n\n"
                "Fix: Add Part → 30+ photos → annotate → Train AI\n\n"
                "Open camera anyway (testing only)?"): return
        self.cap=cv2.VideoCapture(0)
        if not self.cap.isOpened(): messagebox.showerror("Camera","Cannot open camera."); return
        self.running=True; self.btn_start.config(state='disabled'); self.btn_stop.config(state='normal')
        sc=f"● Running  |  {self.detector.model_info}" if self.detector.is_ready() else "● Running  (No Model)"
        self.sys_lbl.config(text=sc,fg=GREEN if self.detector.is_ready() else ORANGE)
        threading.Thread(target=self._loop,daemon=True).start()

    def stop_camera(self):
        self.running=False
        if self.cap: self.cap.release(); self.cap=None
        self.btn_start.config(state='normal'); self.btn_stop.config(state='disabled')
        self.sys_lbl.config(text="● Stopped",fg=YELLOW)
        self.canvas.delete('all'); self.det_lbl.config(text="Objects Detected : 0"); self.fps_lbl.config(text="FPS : 0.0")
        self._sim.clear()

    def _loop(self):
        while self.running and self.cap:
            ret,frame=self.cap.read()
            if not ret: break
            h,w=frame.shape[:2]
            dets=self.detector.detect(frame)
            frame,overlay=self.detector.draw_overlay(frame,dets,get_part_by_yolo_class)
            self._overlay=overlay
            cv2.putText(frame,f"FPS:{self.detector.last_fps}",(10,h-12),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,100),2)
            if not self.detector.is_ready():
                cv2.putText(frame,"NO TRAINED MODEL",(10,36),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,80,255),2)
            self.det_lbl.config(text=f"Objects Detected : {len(dets)}")
            self.fps_lbl.config(text=f"FPS : {self.detector.last_fps}")

            # ── Update detected-parts panel (throttled by label change check inside panel) ──
            self.after(0, lambda ov=list(overlay): self._sim.update_detections(ov))

            # ── Log detections only (no auto-select) ──────────────────────────
            for item in overlay:
                p=item.get('part')
                if p:
                    pno=p['part_no']; now=time.time()
                    if now-self._last_logged.get(pno,0)>10:
                        self._last_logged[pno]=now; log_detection(pno,p['part_name'],item['conf'],'',p.get('judgement',''))

            cw=self.canvas.winfo_width() or 860; ch=self.canvas.winfo_height() or 500
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            img=Image.fromarray(rgb).resize((cw,ch),Image.BILINEAR)
            photo=ImageTk.PhotoImage(img); self.canvas.create_image(0,0,anchor='nw',image=photo); self.canvas.image=photo

    def _on_click(self,event):
        cw=self.canvas.winfo_width() or 860; ch=self.canvas.winfo_height() or 500
        for item in self._overlay:
            x1,y1,x2,y2=item['bbox']
            sx1=int(x1*cw/640); sy1=int(y1*ch/480); sx2=int(x2*cw/640); sy2=int(y2*ch/480)
            if sx1<=event.x<=sx2 and sy1<=event.y<=sy2:
                if item['part']: self._last_det_pno=None; self._select_part(item['part'])
                else: messagebox.showinfo("Unknown Part",f"'{item['label']}' not in database.\nAdd it via Add Part.")
                return

    def _select_part(self,p):
        self._selected=p
        for key,(var,lv,dcol) in self._dvars.items():
            val=str(p.get(key,'') or '') or '—'; var.set(val)
            lv.config(fg=dcol)
        imgs=get_part_images(p['part_no'])
        path=p.get('image_path','') or (imgs[0]['image_path'] if imgs else '')
        if path and os.path.exists(path):
            try:
                img=Image.open(path).convert('RGB')
                # Scale proportionally to fill the full panel width
                panel_w=self.part_img.winfo_width() or 414
                ow,oh=img.size
                if ow>0:
                    nh=max(1,int(oh*panel_w/ow))
                    img=img.resize((panel_w,nh),Image.LANCZOS)
                ph=ImageTk.PhotoImage(img); self.part_img.config(image=ph,text=''); self.part_img.image=ph
            except Exception: pass
        else:
            self.part_img.config(image='',text='No image',fg=MUTED); self.part_img.image=None

    def _on_sim_click(self, part):
        """Called when user taps a card — instantly loads Part Information."""
        self._select_part(part)

    def take_screenshot(self):
        if not self.cap: messagebox.showinfo("","Start camera first."); return
        ret,frame=self.cap.read()
        if ret:
            ts=datetime.now().strftime('%Y%m%d_%H%M%S')
            p=os.path.join(SCREENS_DIR,f'capture_{ts}.png'); cv2.imwrite(p,frame)
            messagebox.showinfo("Saved",f"Screenshot:\n{p}")

    def _export(self):
        if not self._selected: messagebox.showinfo("","Tap a detected part first."); return
        p=self._selected
        path=filedialog.asksaveasfilename(defaultextension='.csv',
            initialfile=f"part_{p['part_no']}.csv",filetypes=[("CSV","*.csv")])
        if path:
            with open(path,'w',newline='') as f:
                w=csv.writer(f); w.writerow(['Field','Value'])
                for k,v in p.items(): w.writerow([k,v])
            messagebox.showinfo("Exported",f"Saved:\n{path}")


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE SCREEN
# ══════════════════════════════════════════════════════════════════════════════
class DatabaseScreen(tk.Frame):
    def __init__(self,parent):
        super().__init__(parent,bg=BG); self._build(); self._load()

    def _build(self):
        hdr=tk.Frame(self,bg=HDR,pady=10); hdr.pack(fill='x')
        tk.Label(hdr,text="🗄️  Parts Database",bg=HDR,fg='white',font=FT).pack(side='left',padx=16)
        TB(hdr,"➕ Add New Part",self.master._show_add,bg=GREEN,fg='#000',font=FH3,padx=16).pack(side='right',padx=16)
        sf=tk.Frame(self,bg=BG,padx=14,pady=8); sf.pack(fill='x')
        tk.Label(sf,text="🔍",bg=BG,fg=MUTED,font=FB).pack(side='left')
        self.sv=tk.StringVar(); self.sv.trace('w',lambda*_:self._load())
        se=tk.Entry(sf,textvariable=self.sv,bg=CARD,fg=TEXT,insertbackground=TEXT,font=FB,
                    relief='flat',bd=0,width=36,highlightthickness=1,highlightbackground=BORDER)
        se.pack(side='left',padx=8,ipady=8)
        attach_kb(self.winfo_toplevel(),se)
        self.cnt=tk.Label(sf,text="",bg=BG,fg=MUTED,font=FSM); self.cnt.pack(side='right')
        pane=tk.Frame(self,bg=BG); pane.pack(fill='both',expand=True,padx=14,pady=8)
        sty=ttk.Style()
        sty.configure('DB.Treeview',background=CARD,foreground=TEXT,fieldbackground=CARD,rowheight=36,font=FB)
        sty.configure('DB.Treeview.Heading',background=HDR,foreground='white',font=FH3)
        sty.map('DB.Treeview',background=[('selected',BLU)])
        cols=('part_no','part_name','model','yolo_class','judgement','qty')
        self.tree=ttk.Treeview(pane,columns=cols,show='headings',style='DB.Treeview')
        for c,h,w in [('part_no','Part No',130),('part_name','Part Name',210),('model','Model',90),
                      ('yolo_class','YOLO Class',130),('judgement','Judgement',110),('qty','Qty',60)]:
            self.tree.heading(c,text=h); self.tree.column(c,width=w)
        sb2=ttk.Scrollbar(pane,orient='vertical',command=self.tree.yview); self.tree.configure(yscrollcommand=sb2.set)
        self.tree.pack(side='left',fill='both',expand=True); sb2.pack(side='left',fill='y')
        self.tree.bind('<Double-1>',self._edit); self.tree.bind('<<TreeviewSelect>>',self._preview)
        rp=tk.Frame(pane,bg=CARD,width=250,padx=12,pady=12); rp.pack(side='right',fill='y',padx=(8,0)); rp.pack_propagate(False)
        tk.Label(rp,text="Part Photo",bg=CARD,fg=CYAN,font=FH3).pack(anchor='w')
        self.dbi=tk.Label(rp,bg=PANEL,fg=MUTED,text="Select part",font=FSM,width=26,height=9); self.dbi.pack(fill='x',pady=8)
        TB(rp,"✏️ Edit",lambda:self._edit(None),bg=BLU,fg='white',font=FB).pack(fill='x',pady=3)
        TB(rp,"🗑 Delete",self._delete,bg=DRED,fg='white',font=FB).pack(fill='x',pady=3)
        TB(rp,"📋 Export CSV",self._export_csv,bg=TEAL,fg='white',font=FB).pack(fill='x',pady=3)

    def _load(self):
        q=self.sv.get().strip(); parts=search_parts(q) if q else get_all_parts()
        for r in self.tree.get_children(): self.tree.delete(r)
        for p in parts: self.tree.insert('','end',iid=p['part_no'],
            values=(p['part_no'],p['part_name'],p.get('model',''),p.get('yolo_class',''),p.get('judgement',''),p.get('quantity','')))
        self.cnt.config(text=f"{len(parts)} part(s)")

    def _preview(self,e):
        sel=self.tree.selection()
        if not sel: return
        p=get_part_by_no(sel[0])
        if not p: return
        imgs=get_part_images(p['part_no']); path=p.get('image_path','') or (imgs[0]['image_path'] if imgs else '')
        if path and os.path.exists(path):
            try:
                img=Image.open(path).convert('RGB'); img.thumbnail((220,140),Image.LANCZOS)
                ph=ImageTk.PhotoImage(img); self.dbi.config(image=ph,text=''); self.dbi.image=ph
            except Exception: pass
        else: self.dbi.config(image='',text='No photo',fg=MUTED); self.dbi.image=None

    def _edit(self,e):
        sel=self.tree.selection()
        if not sel: return
        p=get_part_by_no(sel[0])
        if p: self.master.show_edit(p)

    def _delete(self):
        sel=self.tree.selection()
        if not sel: return
        if messagebox.askyesno("Delete",f"Delete {len(sel)} part(s)?"):
            for pno in sel: delete_part(pno)
            self._load()

    def _export_csv(self):
        parts=get_all_parts()
        if not parts: messagebox.showinfo("","No parts."); return
        path=filedialog.asksaveasfilename(defaultextension='.csv',initialfile='parts.csv',filetypes=[("CSV","*.csv")])
        if path:
            with open(path,'w',newline='') as f:
                w=csv.writer(f); w.writerow(parts[0].keys())
                for p in parts: w.writerow(p.values())
            messagebox.showinfo("Exported",f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  HISTORY SCREEN
# ══════════════════════════════════════════════════════════════════════════════
class HistoryScreen(tk.Frame):
    def __init__(self,parent):
        super().__init__(parent,bg=BG); self._build(); self._load()

    def _build(self):
        hdr=tk.Frame(self,bg=HDR,pady=10); hdr.pack(fill='x')
        tk.Label(hdr,text="📜  Detection History",bg=HDR,fg='white',font=FT).pack(side='left',padx=16)
        TB(hdr,"🗑 Clear All",self._clear,bg=DRED,fg='white',font=FB,padx=14).pack(side='right',padx=16)
        sty=ttk.Style()
        sty.configure('H.Treeview',background=CARD,foreground=TEXT,fieldbackground=CARD,rowheight=36,font=FB)
        sty.configure('H.Treeview.Heading',background=HDR,foreground='white',font=FH3)
        sty.map('H.Treeview',background=[('selected',BLU)])
        cols=('ts','pno','pname','conf','judge')
        self.tree=ttk.Treeview(self,columns=cols,show='headings',style='H.Treeview')
        for c,h,w in [('ts','Timestamp',165),('pno','Part No',140),('pname','Part Name',210),('conf','Confidence',110),('judge','Judgement',130)]:
            self.tree.heading(c,text=h); self.tree.column(c,width=w)
        sb=ttk.Scrollbar(self,orient='vertical',command=self.tree.yview); self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side='left',fill='both',expand=True,padx=(14,0),pady=8); sb.pack(side='left',fill='y',pady=8)

    def _load(self):
        rows=get_history(500)
        for r in self.tree.get_children(): self.tree.delete(r)
        for h in rows:
            conf=f"{int(h['confidence']*100)}%" if h['confidence'] else '—'
            self.tree.insert('','end',values=(h['timestamp'],h['part_no'],h['part_name'],conf,h['judgement']))

    def _clear(self):
        if messagebox.askyesno("Clear","Clear all history?"): clear_history(); self._load()


# ══════════════════════════════════════════════════════════════════════════════
#  APP SHELL
# ══════════════════════════════════════════════════════════════════════════════
class AutoPartApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Based Part Identification System")
        self.geometry("1366x768")    # common tablet resolution
        self.minsize(1024, 640)
        self.configure(bg=BG)
        # Tablet: maximise on start
        try: self.state('zoomed')
        except Exception:
            try: self.attributes('-zoomed', True)
            except Exception: pass
        init_db()
        self._screens = {}
        setup_global_mousewheel(self)   # mouse wheel works anywhere on screen
        self._build_chrome()
        self._show_main()

    def _build_chrome(self):
        top=tk.Frame(self,bg=PANEL,height=66); top.pack(fill='x'); top.pack_propagate(False)
        tk.Label(top,text="⚙️",bg=PANEL,fg=CYAN,font=("Arial",28)).pack(side='left',padx=12)
        tk.Label(top,text="AI BASED PART IDENTIFICATION SYSTEM",bg=PANEL,fg=TEXT,font=("Arial",16,"bold")).pack(side='left',padx=4)
        self.clk=tk.Label(top,bg=PANEL,fg=TEXT,font=("Arial",12,"bold"),anchor='e'); self.clk.pack(side='right',padx=18)
        self._tick()
        tk.Frame(self,bg=BORDER,height=1).pack(fill='x')
        menu=tk.Frame(self,bg=CARD,pady=4); menu.pack(fill='x')
        for lbl,cmd in [("🏠 Home",self._show_main),("➕ Add Part",self._show_add),
                        ("🗄️ Database",self._show_db),("📜 History",self._show_history),
                        ("🧠 Train AI",lambda:TrainingWindow(self))]:
            tk.Button(menu,text=lbl,command=cmd,bg=CARD,fg=TEXT,font=FB,relief='flat',
                      padx=18,pady=8,cursor='hand2',activebackground=BORDER,
                      activeforeground=TEXT).pack(side='left',padx=2)
        tk.Frame(self,bg=BORDER,height=1).pack(fill='x')

    def _tick(self):
        now=datetime.now(); self.clk.config(text=f"{now.strftime('%d %b %Y')}\n{now.strftime('%I:%M:%S %p')}"); self.after(1000,self._tick)

    def _clear(self):
        for w in self._screens.values(): w.pack_forget()

    def _show_main(self):
        self._clear()
        if 'main' not in self._screens: self._screens['main']=MainDetectionScreen(self)
        self._screens['main'].pack(fill='both',expand=True)

    def _show_add(self):
        self._clear()
        if 'add' in self._screens: self._screens['add'].destroy()
        self._screens['add']=AddPartScreen(self)
        self._screens['add'].pack(fill='both',expand=True)

    def _show_db(self):
        self._clear()
        if 'db' in self._screens: self._screens['db'].destroy()
        self._screens['db']=DatabaseScreen(self)
        self._screens['db'].pack(fill='both',expand=True)

    def _show_history(self):
        self._clear()
        if 'hist' in self._screens: self._screens['hist'].destroy()
        self._screens['hist']=HistoryScreen(self)
        self._screens['hist'].pack(fill='both',expand=True)

    def show_edit(self,part):
        self._clear()
        if 'add' in self._screens: self._screens['add'].destroy()
        self._screens['add']=AddPartScreen(self,existing=part)
        self._screens['add'].pack(fill='both',expand=True)

    def on_close(self):
        if 'main' in self._screens: self._screens['main'].stop_camera()
        self.destroy()


def main():
    app=AutoPartApp()
    app.protocol("WM_DELETE_WINDOW",app.on_close)
    app.mainloop()

if __name__=='__main__': main()
