"""
tablet_utils.py
===============
Tablet/touch enhancements:
  1. Virtual on-screen keyboard — pops when Entry is tapped
  2. Global mouse-wheel scroll — wheel works anywhere on screen, scrolls active canvas
  3. Touch-scroll — swipe anywhere on canvas to scroll
  4. Offline YOLO model loader
"""
import tkinter as tk
from tkinter import ttk
import os, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Keyboard colours ──────────────────────────────────────────────────────────
KB_BG  = "#0a1628"
KB_KEY = "#1e3a5f"
KB_TXT = "#e0e8f0"
KB_ACT = "#1565c0"
KB_SP  = "#37474f"
KB_DEL = "#c62828"
KB_CAP = "#00838f"

ROWS = [
    list("1234567890-_"),
    list("qwertyuiop"),
    list("asdfghjkl"),
    list("zxcvbnm."),
]

MODEL_URLS = {
    'yolov8n.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt',
    'yolov8s.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt',
    'yolov8m.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt',
}

PLACEHOLDER_COLOR = '#6a8aaa'   # MUTED
TEXT_COLOR        = '#e0e8f0'   # TEXT


# ══════════════════════════════════════════════════════════════════════════════
#  VIRTUAL KEYBOARD
# ══════════════════════════════════════════════════════════════════════════════
class VirtualKeyboard(tk.Toplevel):
    """
    Floating on-screen keyboard.
    FIX: ✖ close now uses withdraw()+explicit destroy, not just destroy(),
         so overrideredirect windows close correctly.
    FIX: Types into entry directly, skipping placeholder.
    """
    _inst = None   # singleton — only one keyboard at a time

    @classmethod
    def show(cls, root, entry):
        if cls._inst is not None:
            try:
                if cls._inst.winfo_exists():
                    cls._inst._attach(entry)
                    return
            except Exception:
                pass
        cls._inst = cls(root, entry)

    @classmethod
    def close_current(cls):
        if cls._inst is not None:
            try:
                if cls._inst.winfo_exists():
                    cls._inst._close()
            except Exception:
                pass
            cls._inst = None

    def __init__(self, root, entry):
        super().__init__(root)
        self.overrideredirect(True)      # no title bar
        self.attributes('-topmost', True)
        self.configure(bg=KB_BG)
        self._root  = root
        self._entry = entry
        self._caps  = False
        self._btns  = {}
        self._build()
        self._pos()
        # Detect taps outside keyboard
        root.bind_all('<ButtonPress-1>', self._maybe_close, add='+')

    # ── Build ─────────────────────────────────────────────────────────────────
    def _build(self):
        wrap = tk.Frame(self, bg=KB_BG, padx=6, pady=6)
        wrap.pack()

        for row_chars in ROWS:
            row = tk.Frame(wrap, bg=KB_BG)
            row.pack(pady=2)
            for ch in row_chars:
                b = tk.Button(row, text=ch, width=3, height=1,
                              bg=KB_KEY, fg=KB_TXT,
                              font=("Arial", 14, "bold"),
                              relief='flat', bd=0,
                              activebackground=KB_ACT,
                              activeforeground='white',
                              command=lambda c=ch: self._type(c))
                b.pack(side='left', padx=2)
                self._btns[ch] = b

        # Special keys row
        spec = tk.Frame(wrap, bg=KB_BG)
        spec.pack(pady=2)

        self._cap_btn = tk.Button(spec, text="CAPS", width=6, height=1,
                                  bg=KB_CAP, fg='white',
                                  font=("Arial", 12, "bold"), relief='flat',
                                  command=self._toggle_caps)
        self._cap_btn.pack(side='left', padx=2)

        tk.Button(spec, text="    SPACE    ", height=1,
                  bg=KB_SP, fg=KB_TXT, font=("Arial", 13),
                  relief='flat',
                  command=lambda: self._type(' ')).pack(side='left', padx=2)

        tk.Button(spec, text="⌫", width=4, height=1,
                  bg=KB_DEL, fg='white',
                  font=("Arial", 14, "bold"), relief='flat',
                  command=self._back).pack(side='left', padx=2)

        # ── Close button — uses _close() not destroy() directly ──
        close_btn = tk.Button(spec, text="✖", width=4, height=1,
                              bg=KB_SP, fg=KB_TXT,
                              font=("Arial", 13), relief='flat',
                              command=self._close)
        close_btn.pack(side='left', padx=2)

    def _close(self):
        """Close keyboard and deselect/unfocus the active entry."""
        VirtualKeyboard._inst = None
        # Remove the blue highlight from the active entry
        try:
            if self._entry and self._entry.winfo_exists():
                self._entry.config(highlightbackground='#1e3a5f',
                                   highlightcolor='#1e3a5f')
                # Move focus away from the entry
                self._root.focus_set()
                self._entry.select_clear()
        except Exception:
            pass
        try:
            self._root.unbind_all('<ButtonPress-1>')
        except Exception:
            pass
        try:
            self.withdraw()
            self.destroy()
        except Exception:
            pass

    # ── Typing ────────────────────────────────────────────────────────────────
    def _is_placeholder(self):
        """Returns True if the entry is still showing placeholder text."""
        try:
            return self._entry.cget('fg') == PLACEHOLDER_COLOR
        except Exception:
            return False

    def _clear_placeholder(self):
        """Remove placeholder text and set normal text colour."""
        try:
            self._entry.delete(0, tk.END)
            self._entry.config(fg=TEXT_COLOR)
        except Exception:
            pass

    def _type(self, ch):
        if not self._entry or not self._entry.winfo_exists():
            return
        if self._is_placeholder():
            self._clear_placeholder()
        char = ch.upper() if self._caps else ch
        self._entry.insert(tk.INSERT, char)
        self._entry.focus_set()

    def _back(self):
        if not self._entry or not self._entry.winfo_exists():
            return
        if self._is_placeholder():
            self._clear_placeholder()
            self._entry.focus_set()
            return
        val = self._entry.get()
        if val:
            self._entry.delete(len(val) - 1, tk.END)
        self._entry.focus_set()

    def _toggle_caps(self):
        self._caps = not self._caps
        self._cap_btn.config(bg=KB_ACT if self._caps else KB_CAP)
        for ch, btn in self._btns.items():
            btn.config(text=ch.upper() if self._caps else ch)

    def _attach(self, entry):
        self._entry = entry
        self._pos()

    def _pos(self):
        """Position keyboard below the entry, or above if no room."""
        self.update_idletasks()
        try:
            ex = self._entry.winfo_rootx()
            ey = self._entry.winfo_rooty() + self._entry.winfo_height() + 4
            kw = self.winfo_reqwidth()
            kh = self.winfo_reqheight()
            sw = self.winfo_screenwidth()
            sh = self.winfo_screenheight()
            x  = max(0, min(ex, sw - kw))
            y  = ey if (ey + kh) < sh else max(0, ey - kh - self._entry.winfo_height() - 8)
            self.geometry(f"+{x}+{y}")
        except Exception:
            pass

    def _maybe_close(self, event):
        """Close if user tapped outside the keyboard window."""
        try:
            if not self.winfo_exists():
                return
            wx = self.winfo_rootx();  wy = self.winfo_rooty()
            ww = self.winfo_width();  wh = self.winfo_height()
            # Inside keyboard — keep open
            if wx <= event.x_root <= wx + ww and wy <= event.y_root <= wy + wh:
                return
            # Tapped another Entry — switch to it
            w = event.widget
            if isinstance(w, tk.Entry):
                self._attach(w)
                return
            # Tapped elsewhere — close
            self._close()
        except Exception:
            pass


def attach_kb(root, entry):
    """Bind virtual keyboard to one Entry widget."""
    def _open(e):
        VirtualKeyboard.show(root, entry)
    entry.bind('<FocusIn>',      _open, add='+')
    entry.bind('<ButtonPress-1>', _open, add='+')


def attach_kb_all(root, container):
    """Recursively attach virtual keyboard to every Entry in a frame tree."""
    for w in container.winfo_children():
        if isinstance(w, tk.Entry):
            attach_kb(root, w)
        else:
            try:
                attach_kb_all(root, w)
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL MOUSE-WHEEL SCROLL
#  Binds to the root window so the wheel works regardless of which widget
#  the cursor is over. Finds the nearest scrollable canvas ancestor.
# ══════════════════════════════════════════════════════════════════════════════

# Registry of all scrollable canvases (registered via make_touch_scrollable)
_scroll_canvases = []


def register_scroll_canvas(canvas):
    """Call this for every canvas that should be globally scrollable."""
    if canvas not in _scroll_canvases:
        _scroll_canvases.append(canvas)


def setup_global_mousewheel(root):
    """
    Bind mouse-wheel on the root window so scrolling works anywhere —
    even when the cursor is over a label, entry, or frame.
    Scrolls the topmost visible registered canvas.
    """
    def _scroll(event):
        # Find which registered canvases are currently visible
        for cv in reversed(_scroll_canvases):  # last registered = most recent tab
            try:
                if cv.winfo_ismapped() and cv.winfo_viewable():
                    # On Windows event.delta is ±120 per notch.
                    # Positive delta = wheel rolled UP → scroll content UP (negative units)
                    units = -1 if event.delta > 0 else 1
                    cv.yview_scroll(units * 3, 'units')
                    return
            except Exception:
                continue

    # Windows / macOS
    root.bind_all('<MouseWheel>', _scroll, add='+')
    # Linux
    root.bind_all('<Button-4>',
                  lambda e: _scroll_canvas_by(e, -1), add='+')
    root.bind_all('<Button-5>',
                  lambda e: _scroll_canvas_by(e,  1), add='+')


def _scroll_canvas_by(event, direction):
    for cv in reversed(_scroll_canvases):
        try:
            if cv.winfo_ismapped() and cv.winfo_viewable():
                cv.yview_scroll(direction * 3, 'units')
                return
        except Exception:
            continue


# ══════════════════════════════════════════════════════════════════════════════
#  TOUCH SCROLL  — finger swipe on canvas to scroll
# ══════════════════════════════════════════════════════════════════════════════
class TouchScroll:
    """
    Attach to a tk.Canvas. Finger swipe = scroll.
    Distinguishes vertical swipe (scroll) from short tap (click).
    """
    DRAG_THRESHOLD = 8

    def __init__(self, canvas: tk.Canvas):
        self._cv    = canvas
        self._y0    = None
        self._moved = False
        canvas.bind('<ButtonPress-1>',   self._start, add='+')
        canvas.bind('<B1-Motion>',       self._move,  add='+')
        canvas.bind('<ButtonRelease-1>', self._end,   add='+')
        # Also bind MouseWheel directly on the canvas for reliable scrolling
        canvas.bind('<MouseWheel>',
                    lambda e: canvas.yview_scroll(
                        (-1 if e.delta > 0 else 1) * 3, 'units'), add='+')
        canvas.bind('<Button-4>',
                    lambda e: canvas.yview_scroll(-3, 'units'), add='+')
        canvas.bind('<Button-5>',
                    lambda e: canvas.yview_scroll(3, 'units'), add='+')

    def _start(self, e):
        self._y0    = e.y_root
        self._moved = False

    def _move(self, e):
        if self._y0 is None:
            return
        dy = self._y0 - e.y_root
        if abs(dy) > self.DRAG_THRESHOLD:
            self._moved = True
            units = int(dy / 10)
            if units:
                self._cv.yview_scroll(units, 'units')
                self._y0 = e.y_root

    def _end(self, e):
        self._y0 = None


def make_touch_scrollable(parent, bg='#0d1b2e'):
    """
    Creates (canvas, inner_frame) inside parent.
    - Swipe on canvas → scrolls
    - Mouse wheel anywhere on screen → scrolls (via global binding)
    - Scrollbar thumb → scrolls
    Returns (canvas, inner_frame).
    """
    c  = tk.Canvas(parent, bg=bg, highlightthickness=0)
    sb = ttk.Scrollbar(parent, orient='vertical', command=c.yview)
    c.configure(yscrollcommand=sb.set)
    sb.pack(side='right', fill='y')
    c.pack(fill='both', expand=True)

    f   = tk.Frame(c, bg=bg)
    wid = c.create_window((0, 0), window=f, anchor='nw')
    c.bind('<Configure>', lambda e: c.itemconfig(wid, width=e.width))
    f.bind('<Configure>',
           lambda e: c.configure(scrollregion=c.bbox('all')))

    TouchScroll(c)              # finger swipe on canvas + direct mousewheel
    register_scroll_canvas(c)  # register for global mouse-wheel
    return c, f


# ══════════════════════════════════════════════════════════════════════════════
#  OFFLINE MODEL LOADER
# ══════════════════════════════════════════════════════════════════════════════
def ensure_model(name='yolov8s.pt', log=None):
    def _l(msg):
        if log: log(msg)
        print(msg)

    model_dir = os.path.join(ROOT, 'trained_model')
    os.makedirs(model_dir, exist_ok=True)

    candidates = [
        os.path.join(model_dir, name),
        os.path.join(os.getcwd(), name),
        os.path.join(ROOT, name),
        os.path.join(os.path.expanduser('~'), '.ultralytics', 'assets', name),
    ]
    for p in candidates:
        if os.path.exists(p):
            dst = os.path.join(model_dir, name)
            if p != dst:
                import shutil; shutil.copy2(p, dst)
            _l(f"[Model] Using: {dst}")
            return dst

    url = MODEL_URLS.get(name)
    if not url:
        raise FileNotFoundError(f"No URL for '{name}'")
    dst = os.path.join(model_dir, name)
    _l(f"[Model] Downloading {name}...")
    urllib.request.urlretrieve(url, dst)
    _l(f"[Model] Saved to {dst}  ({os.path.getsize(dst)//1024//1024} MB)")
    return dst
