"""
Database Manager — AutoPartDetector
Includes: part_images table, annotation import tracking
"""
import sqlite3, os, shutil
from datetime import datetime

ROOT          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH       = os.path.join(ROOT, 'database', 'parts.db')
PARTS_IMG_DIR = os.path.join(ROOT, 'parts_images')
LABELS_DIR    = os.path.join(ROOT, 'dataset', 'labels_raw')


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(PARTS_IMG_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR,    exist_ok=True)
    conn = get_connection()
    c    = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS parts(
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        part_no      TEXT UNIQUE NOT NULL,
        part_name    TEXT NOT NULL,
        model        TEXT, supplier TEXT, group_name TEXT,
        date         TEXT, zone TEXT, quantity INTEGER DEFAULT 0,
        judgement    TEXT, reason TEXT,
        image_path   TEXT, yolo_class TEXT,
        created_at   TEXT DEFAULT CURRENT_TIMESTAMP)''')

    c.execute('''CREATE TABLE IF NOT EXISTS part_images(
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        part_no      TEXT NOT NULL,
        image_path   TEXT NOT NULL,
        label_path   TEXT DEFAULT '',
        annotated    INTEGER DEFAULT 0,
        source       TEXT DEFAULT 'upload',
        created_at   TEXT DEFAULT CURRENT_TIMESTAMP)''')

    c.execute('''CREATE TABLE IF NOT EXISTS detection_history(
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        part_no      TEXT, part_name TEXT, confidence REAL,
        timestamp    TEXT, screenshot_path TEXT, judgement TEXT)''')

    # seed demo
    c.execute('''INSERT OR IGNORE INTO parts
        (part_no,part_name,model,supplier,group_name,date,zone,
         quantity,judgement,reason,image_path,yolo_class)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?)''',
        ('966626027R','Assist Grip Handle','P1324H',
         'Safety Trim - Plastique','A','06-05-2026','TB',
         3,'Scrap','Damage','','grab_handle'))
    conn.commit()
    conn.close()


# ── Parts CRUD ────────────────────────────────────────────────────────────────
def get_all_parts():
    conn = get_connection()
    rows = conn.execute('SELECT * FROM parts ORDER BY created_at DESC').fetchall()
    conn.close(); return [dict(r) for r in rows]

def get_part_by_no(part_no):
    conn = get_connection()
    row  = conn.execute('SELECT * FROM parts WHERE part_no=?',(part_no,)).fetchone()
    conn.close(); return dict(row) if row else None

def get_part_by_yolo_class(yolo_class):
    conn = get_connection()
    row  = conn.execute('SELECT * FROM parts WHERE yolo_class=?',(yolo_class,)).fetchone()
    conn.close(); return dict(row) if row else None

def add_part(data):
    conn = get_connection()
    conn.execute('''INSERT OR REPLACE INTO parts
        (part_no,part_name,model,supplier,group_name,date,zone,
         quantity,judgement,reason,image_path,yolo_class)
        VALUES(:part_no,:part_name,:model,:supplier,:group_name,:date,:zone,
               :quantity,:judgement,:reason,:image_path,:yolo_class)''', data)
    conn.commit(); conn.close()

def delete_part(part_no):
    conn = get_connection()
    conn.execute('DELETE FROM parts WHERE part_no=?',(part_no,))
    conn.execute('DELETE FROM part_images WHERE part_no=?',(part_no,))
    conn.commit(); conn.close()

def search_parts(q):
    conn = get_connection()
    w    = f'%{q}%'
    rows = conn.execute(
        'SELECT * FROM parts WHERE part_no LIKE ? OR part_name LIKE ? OR model LIKE ?',
        (w,w,w)).fetchall()
    conn.close(); return [dict(r) for r in rows]


# ── Part Images CRUD ──────────────────────────────────────────────────────────
def add_part_image(part_no, image_path, label_path='', annotated=0, source='upload'):
    conn = get_connection()
    conn.execute('''INSERT INTO part_images
        (part_no,image_path,label_path,annotated,source)
        VALUES(?,?,?,?,?)''',
        (part_no, image_path, label_path, annotated, source))
    conn.commit(); conn.close()

def get_part_images(part_no):
    conn = get_connection()
    rows = conn.execute(
        'SELECT * FROM part_images WHERE part_no=? ORDER BY id',(part_no,)).fetchall()
    conn.close(); return [dict(r) for r in rows]

def update_part_image_label(img_id, label_path, annotated=1):
    conn = get_connection()
    conn.execute('UPDATE part_images SET label_path=?,annotated=? WHERE id=?',
                 (label_path, annotated, img_id))
    conn.commit(); conn.close()

def delete_part_image(img_id):
    conn = get_connection()
    row  = conn.execute('SELECT * FROM part_images WHERE id=?',(img_id,)).fetchone()
    conn.close()
    if row:
        r = dict(row)
        try:
            if r['image_path'] and os.path.exists(r['image_path']):
                os.remove(r['image_path'])
        except: pass
        try:
            if r['label_path'] and os.path.exists(r['label_path']):
                os.remove(r['label_path'])
        except: pass
    conn2 = get_connection()
    conn2.execute('DELETE FROM part_images WHERE id=?',(img_id,))
    conn2.commit(); conn2.close()

def get_all_annotated_images():
    conn = get_connection()
    rows = conn.execute('''
        SELECT pi.*, p.yolo_class
        FROM part_images pi JOIN parts p ON pi.part_no=p.part_no
        WHERE pi.annotated=1
          AND pi.label_path!=''
          AND p.yolo_class!=''
          AND p.yolo_class IS NOT NULL''').fetchall()
    conn.close(); return [dict(r) for r in rows]

def get_all_yolo_classes():
    conn = get_connection()
    rows = conn.execute(
        "SELECT DISTINCT yolo_class FROM parts WHERE yolo_class!='' AND yolo_class IS NOT NULL"
    ).fetchall()
    conn.close(); return [r[0] for r in rows]

def count_annotated(part_no):
    conn = get_connection()
    n = conn.execute(
        'SELECT COUNT(*) FROM part_images WHERE part_no=? AND annotated=1',(part_no,)
    ).fetchone()[0]
    conn.close(); return n


# ── Detection History ─────────────────────────────────────────────────────────
def log_detection(part_no, part_name, confidence, screenshot_path='', judgement=''):
    conn = get_connection()
    conn.execute('''INSERT INTO detection_history
        (part_no,part_name,confidence,timestamp,screenshot_path,judgement)
        VALUES(?,?,?,?,?,?)''',
        (part_no, part_name, confidence,
         datetime.now().strftime('%Y-%m-%d %H:%M:%S'), screenshot_path, judgement))
    conn.commit(); conn.close()

def get_history(limit=200):
    conn = get_connection()
    rows = conn.execute(
        'SELECT * FROM detection_history ORDER BY timestamp DESC LIMIT ?',(limit,)
    ).fetchall()
    conn.close(); return [dict(r) for r in rows]

def clear_history():
    conn = get_connection()
    conn.execute('DELETE FROM detection_history')
    conn.commit(); conn.close()
