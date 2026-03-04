import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import cv2
import numpy as np
import os
import time
import sys
import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import winsound
import base64
import mediapipe as mp
from supabase import create_client, Client
import traceback  # For full error stack if needed
import faiss  # For fast similarity search

if getattr(sys, 'frozen', False):
    bundle_dir = sys._MEIPASS
    os.environ["INSIGHTFACE_HOME"] = bundle_dir
    print(f"[BUNDLED] INSIGHTFACE_HOME set to: {bundle_dir}")

# ────────────────────────────────────────────────
# Supabase Config
# ────────────────────────────────────────────────
SUPABASE_URL = "https://crujjurupavknjwdjjmj.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNydWpqdXJ1cGF2a25qd2Rqam1qIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzA5NjI0MTAsImV4cCI6MjA4NjUzODQxMH0.MdQDrEHOyQ0mI6HGX986lNMw5cpj5pfUCnKFh88pnzw"

supabase = None
supabase_connected = False

print("=== Supabase Init ===")
try:
    supabase = create_client(SUPABASE_URL.strip(), SUPABASE_KEY.strip())
    supabase.table("employees").select("emp_code", count="planned").limit(0).execute()
    supabase_connected = True
    print("Supabase connected successfully")
except Exception as e:
    print(f"Supabase failed: {str(e)}")
    messagebox.showerror("Cloud Error", f"Supabase failed:\n{str(e)}\nApp will exit.")
    sys.exit(1)

# ────────────────────────────────────────────────
# Paths & Config
# ────────────────────────────────────────────────
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use writable user-specific folder instead of app directory
# This prevents crashes in Program Files (read-only)
USER_APPDATA = os.getenv('APPDATA')
if USER_APPDATA is None:
    USER_APPDATA = os.path.expanduser('~\\AppData\\Roaming')
APP_DATA_ROOT = os.path.join(USER_APPDATA, 'OntechAttendance')
os.makedirs(APP_DATA_ROOT, exist_ok=True)

DATA_DIR = os.path.join(APP_DATA_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

COLOR_SUCCESS = (0, 255, 120)
COLOR_WARNING = (0, 80, 255)
COLOR_UNKNOWN = (0, 0, 255)
COLOR_CLOUD_OK = (0, 200, 100)

SIMILARITY_THRESHOLD    = 0.20  # lowered for testing
GESTURE_HOLD_SECONDS    = 2.8
MIN_TIME_BETWEEN_ACTIONS = 5.0
SUCCESS_SHOW_SECONDS    = 5.0
PROCESS_EVERY_N_FRAMES  = 8  # Increased for better performance

# ────────────────────────────────────────────────
# Globals
# ────────────────────────────────────────────────
face_db = {}  # code → np.array embedding
employee_info = {}  # code → dict with name, dept, etc.
success_message_start = None
success_message_text = ""
gesture_active_until = 0.0
last_action_time = {}
index = None  # Faiss index
code_list = []  # Corresponding codes for faiss indices

# ────────────────────────────────────────────────
# Lazy-load InsightFace
# ────────────────────────────────────────────────
face_analyzer = None

def get_face_analyzer():
    global face_analyzer
    if face_analyzer is None:
        print("Loading face model... please wait (first time can take 10–40 seconds)")
        try:
            from insightface.app import FaceAnalysis
            face_analyzer = FaceAnalysis(name="buffalo_s", providers=["CUDAExecutionProvider", "CPUExecutionProvider"], allowed_modules=['detection', 'recognition'])
            face_analyzer.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.32)
            print("Face model loaded ✓")
        except Exception as e:
            print(f"InsightFace failed: {e}")
            face_analyzer = None
    return face_analyzer

# ────────────────────────────────────────────────
# Build Faiss Index
# ────────────────────────────────────────────────
def build_faiss_index():
    global index, code_list
    if not face_db:
        return
    embeddings = np.array([normalize(face_db[code]) for code in face_db]).astype('float32')
    code_list = list(face_db.keys())
    index = faiss.IndexFlatIP(512)  # Inner Product for cosine similarity after normalization
    index.add(embeddings)
    print(f"FAISS index built with {len(code_list)} faces")

# ────────────────────────────────────────────────
# Load from Supabase
# ────────────────────────────────────────────────
def load_all_from_supabase():
    global face_db, employee_info, last_sync_time
    face_db = {}
    employee_info = {}

    try:
        # Load employee metadata (no embedding here anymore)
        emp_response = supabase.table("employees").select(
            "emp_code, full_name, department, designation, mobile, notes"
        ).execute()

        for row in emp_response.data:
            code = row["emp_code"]
            employee_info[code] = {
                "full_name": row.get("full_name", code),
                "department": row.get("department", ""),
                "designation": row.get("designation", ""),
                "mobile": row.get("mobile", ""),
                "notes": row.get("notes") or "",
            }

        # Load embeddings from dedicated table
        emb_response = supabase.table("face_embeddings").select(
            "emp_code, embedding_base64"
        ).execute()

        count = 0
        for row in emb_response.data:
            code = row["emp_code"]
            b64_str = row.get("embedding_base64")
            if b64_str:
                try:
                    emb_bytes = base64.b64decode(b64_str)
                    emb_array = np.frombuffer(emb_bytes, dtype=np.float32)
                    actual_len = len(emb_array)

                    print(f"Loaded embedding for {code}: {actual_len} floats")

                    if actual_len == 512:
                        face_db[code] = emb_array
                        count += 1
                        print(f"→ VALID 512-dim embedding loaded for {code}")
                    else:
                        print(f"→ REJECTED {code}: wrong size {actual_len} (expected 512)")

                except Exception as e:
                    print(f"→ FAILED decoding {code}: {str(e)}")

        last_sync_time = datetime.datetime.now()
        print(f"\nSummary: {len(employee_info)} employees, {count} valid 512-dim embeddings")
        if count == 0 and len(employee_info) > 0:
            print("WARNING: Employees exist but no valid embeddings loaded")
        
        # Build faiss index after loading
        build_faiss_index()
        
        return True

    except Exception as e:
        print(f"Load failed: {e}")
        messagebox.showerror("Sync Error", str(e))
        return False

# ────────────────────────────────────────────────
# Attendance (Supabase only)
# ────────────────────────────────────────────────
def mark_present(emp_code: str) -> bool:
    emp_code = emp_code.strip().upper()
    today = datetime.date.today().isoformat()
    now_time = datetime.datetime.now().strftime("%H:%M:%S")

    try:
        existing = supabase.table("attendance")\
            .select("id")\
            .eq("emp_code", emp_code)\
            .eq("checkin_date", today)\
            .execute()

        if existing.data:
            print(f"{emp_code} already checked in")
            return False

        supabase.table("attendance").insert({
            "emp_code": emp_code,
            "checkin_date": today,
            "checkin_time": now_time
        }).execute()
        print(f"Check-in: {emp_code}")
        return True
    except Exception as e:
        print(f"Check-in failed: {e}")
        return False

def mark_out(emp_code: str) -> bool:
    emp_code = emp_code.strip().upper()
    today = datetime.date.today().isoformat()
    now_time = datetime.datetime.now().strftime("%H:%M:%S")

    try:
        response = supabase.table("attendance")\
            .select("id, checkout_time")\
            .eq("emp_code", emp_code)\
            .eq("checkin_date", today)\
            .order("id", desc=True)\
            .limit(1)\
            .execute()

        if not response.data or response.data[0]["checkout_time"]:
            print(f"No open check-in for {emp_code}")
            return False

        record_id = response.data[0]["id"]
        supabase.table("attendance")\
            .update({"checkout_time": now_time})\
            .eq("id", record_id)\
            .execute()
        print(f"Check-out: {emp_code}")
        return True
    except Exception as e:
        print(f"Check-out failed: {e}")
        return False

# ────────────────────────────────────────────────
# Utils
# ────────────────────────────────────────────────
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def cosine_similarity(a, b):
    # Trim to shortest length
    min_len = min(len(a), len(b))
    a_trim = a[:min_len]
    b_trim = b[:min_len]

    # Normalize both
    a_norm = normalize(a_trim)
    b_norm = normalize(b_trim)

    # Dot product
    dot = np.dot(a_norm, b_norm)

    # Safety
    norm_prod = np.linalg.norm(a_norm) * np.linalg.norm(b_norm)
    if norm_prod < 1e-8:
        return 0.0

    sc = dot / norm_prod
    print(f"Cosine: {sc:.4f} (dot={dot:.4f}, norm_prod={norm_prod:.4f})")
    return float(sc)

def is_victory_gesture(lm):
    if not lm:
        return False
    return (
        lm[8].y  < lm[6].y  - 0.018 and
        lm[12].y < lm[10].y - 0.018 and
        lm[16].y > lm[14].y + 0.008 and
        lm[20].y > lm[18].y + 0.008
    )

# ────────────────────────────────────────────────
# Splash screen
# ────────────────────────────────────────────────
def show_splash():
    logo_path = os.path.join(BASE_DIR, "OnTech.png")
    if not os.path.exists(logo_path):
        print("Logo not found — skipping")
        return True

    logo = cv2.imread(logo_path)
    if logo is None:
        print("Logo load failed")
        return True

    h, w = logo.shape[:2]
    win_name = "Ontech Face Recognition"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)

    bg = np.zeros((720, 1280, 3), dtype=np.uint8)
    x_offset = (1280 - w) // 2
    y_offset = (720 - h) // 2
    bg[y_offset:y_offset+h, x_offset:x_offset+w] = logo

    for alpha in range(0, 256, 6):
        blended = cv2.addWeighted(bg, alpha/255.0, np.zeros_like(bg), 1 - alpha/255.0, 0)
        cv2.imshow(win_name, blended)
        cv2.waitKey(25)

    cv2.waitKey(1500)

    for alpha in range(255, -1, 8):
        blended = cv2.addWeighted(bg, alpha/255.0, np.zeros_like(bg), 1 - alpha/255.0, 0)
        cv2.imshow(win_name, blended)
        cv2.waitKey(20)

    cv2.destroyAllWindows()
    return True

# ────────────────────────────────────────────────
# UI helpers
# ────────────────────────────────────────────────
def show_employee_list():
    top = tk.Toplevel()
    top.title("Registered Employees (Cloud)")
    top.geometry("1100x700")
    top.minsize(1000, 600)

    tree = ttk.Treeview(top, columns=("Code","Name","Dept","Desig","Mobile","Notes"), show="headings")
    tree.heading("Code", text="Code")
    tree.heading("Name", text="Name")
    tree.heading("Dept", text="Department")
    tree.heading("Desig", text="Designation")
    tree.heading("Mobile", text="Mobile")
    tree.heading("Notes", text="Notes")

    tree.column("Code", width=90, anchor="center")
    tree.column("Name", width=200)
    tree.column("Dept", width=140)
    tree.column("Desig", width=160)
    tree.column("Mobile", width=120)
    tree.column("Notes", width=250)

    tree.pack(fill="both", expand=True, padx=10, pady=10)

    sb = ttk.Scrollbar(top, orient="vertical", command=tree.yview)
    tree.configure(yscroll=sb.set)
    sb.pack(side="right", fill="y")

    for code, info in employee_info.items():
        notes = info.get("notes") or ""
        display_notes = notes[:90] + "…" if len(notes) > 90 else notes

        tree.insert("", "end", values=(
            code,
            info["full_name"],
            info["department"],
            info["designation"],
            info["mobile"],
            display_notes
        ))

    if not employee_info:
        tree.insert("", "end", values=("", "No employees loaded from cloud", "", "", "", ""))

def show_today_attendance():
    top = tk.Toplevel()
    top.title("Today's Attendance (Cloud)")
    top.geometry("900x600")

    tree = ttk.Treeview(top, columns=("Code","Name","Dept","Check-in","Check-out"), show="headings")
    tree.heading("Code", text="Code")
    tree.heading("Name", text="Name")
    tree.heading("Dept", text="Department")
    tree.heading("Check-in", text="Check-in")
    tree.heading("Check-out", text="Check-out")

    tree.column("Code", width=100, anchor="center")
    tree.column("Name", width=220)
    tree.column("Dept", width=180)
    tree.column("Check-in", width=140)
    tree.column("Check-out", width=140)

    tree.pack(fill="both", expand=True, padx=10, pady=10)

    sb = ttk.Scrollbar(top, orient="vertical", command=tree.yview)
    tree.configure(yscroll=sb.set)
    sb.pack(side="right", fill="y")

    try:
        today = datetime.date.today().isoformat()
        records = supabase.table("attendance")\
            .select("emp_code, checkin_time, checkout_time")\
            .eq("checkin_date", today)\
            .execute().data

        for r in records:
            code = r["emp_code"]
            name = employee_info.get(code, {}).get("full_name", code)
            dept = employee_info.get(code, {}).get("department", "")
            tree.insert("", "end", values=(
                code, name, dept,
                r["checkin_time"] or "-",
                r["checkout_time"] or "-"
            ))

        if not records:
            tree.insert("", "end", values=("", "No attendance today", "", "", ""))
    except Exception as e:
        tree.insert("", "end", values=("", f"Error: {str(e)}", "", "", ""))

# ────────────────────────────────────────────────
# Recognition loop
# ────────────────────────────────────────────────
def run_attendance_recognition():
    print("Available cameras:")
    for i in range(5):
        c = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if c.isOpened():
            print(f"→ Camera {i} works")
            c.release()
        else:
            print(f"→ Camera {i} FAILED")
            
    global success_message_start, success_message_text, gesture_active_until

    analyzer = get_face_analyzer()
    if analyzer is None:
        messagebox.showerror("Error", "Cannot load face model.")
        return

    if not load_all_from_supabase():
        messagebox.showwarning("Warning", "Failed to load faces from cloud.")

    if not face_db:
        messagebox.showwarning("No Faces", "No valid embeddings loaded.\nOnly 'Unknown' will be detected.")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("CAP_DSHOW failed → trying plain index 0")
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("CAP_DSHOW failed → trying index 1")
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("CAP_DSHOW failed → trying index 2")
        cap = cv2.VideoCapture(2)

    print(f"Final cap opened: {cap.isOpened()}")
    print(f"Backend: {cap.getBackendName() if hasattr(cap, 'getBackendName') else 'unknown'}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    WINDOW_NAME = "Ontech Attendance"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.52,
        min_tracking_confidence=0.52
    )

    frame_count = 0
    last_results = []
    prev_time = time.time()

    print("Started ")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame lost → retrying...")
            cap.release()
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)
            time.sleep(0.5)
            continue

        display_frame = frame.copy()
        frame_count += 1
        now = time.time()

        # Gesture detection
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(rgb)
            if hand_results.multi_hand_landmarks:
                for hlm in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        display_frame, hlm, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    if is_victory_gesture(hlm.landmark):
                        gesture_active_until = now + GESTURE_HOLD_SECONDS
                        winsound.Beep(1800, 80)
        except Exception as e:
            print(f"Hand error: {e}")

        # Face processing
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            try:
                faces = analyzer.get(frame)
                results = []

                for face in faces:
                    if face.det_score < 0.30:
                        continue

                    emb = normalize(face.embedding).reshape(1, -1).astype('float32')
                    print(f"Live embedding - shape: {emb.shape}, norm: {np.linalg.norm(emb):.6f}, first 5: {emb[0][:5]}")
                    best_code = "Unknown"
                    best_score = -1.0

                    if index is not None:
                        D, I = index.search(emb, 1)  # Get top 1 match
                        if len(I[0]) > 0 and D[0][0] >= SIMILARITY_THRESHOLD:
                            best_idx = I[0][0]
                            best_code = code_list[best_idx]
                            best_score = D[0][0]
                        print(f"FAISS best similarity: {best_score:.3f} to {best_code}")

                    info = employee_info.get(best_code, {"full_name": best_code})
                    display_name = f"{info['full_name']} " if info['department'] else info['full_name']
                    bbox = face.bbox.astype(int)
                    
                    # FIXED: use tuple () instead of set {}
                    results.append((bbox, display_name, best_score, face.det_score, best_code))

                last_results = results

            except Exception as e:
                print(f"[FACE CRASH] {str(e)}")
                print(traceback.format_exc())
                last_results = []

        # Draw & action
        for bbox, dname, score, det_conf, code in last_results:
            x1,y1,x2,y2 = map(int, bbox)
            color = COLOR_SUCCESS if code != "Unknown" and score >= SIMILARITY_THRESHOLD else COLOR_UNKNOWN

            cv2.rectangle(display_frame, (x1,y1), (x2,y2), color, 3)

            label = f"{dname} {score:.3f}"
            if code == "Unknown":
                label += f" det:{det_conf:.2f}"
            else:
                label += f" sim:{score:.3f}"

            tw = len(label) * 11 + 20
            cv2.rectangle(display_frame, (x1, y1-35), (x1 + tw, y1-5), color, -1)
            cv2.putText(display_frame, label, (x1+8, y1-12),
                        cv2.FONT_HERSHEY_DUPLEX, 0.85, (0,0,0), 2)

            if code != "Unknown" and score >= SIMILARITY_THRESHOLD:
                if code in last_action_time and now - last_action_time[code] < MIN_TIME_BETWEEN_ACTIONS:
                    continue

                is_checkout = now < gesture_active_until
                success = False
                action_text = ""

                if is_checkout:
                    success = mark_out(code)
                    action_text = "CHECKED OUT"
                else:
                    success = mark_present(code)
                    action_text = "CHECKED IN"

                if success:
                    last_action_time[code] = now
                    winsound.Beep(1200, 400)
                    success_message_text = f"{action_text} → {dname.split(' (')[0]}"
                    success_message_start = now
                    print(f"SUCCESS: {action_text} {code}")

        # Success overlay
        if success_message_start and now - success_message_start < SUCCESS_SHOW_SECONDS:
            cv2.putText(display_frame, success_message_text, (140, 160),
                        cv2.FONT_HERSHEY_DUPLEX, 2.2, COLOR_SUCCESS, 6)
            cv2.putText(display_frame, datetime.datetime.now().strftime("%H:%M:%S"),
                        (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.3, COLOR_SUCCESS, 3)

        # Status
        status_text = f"Loaded {len(face_db)} embeddings from cloud"
        status_color = COLOR_CLOUD_OK
        if last_sync_time:
            ago = (datetime.datetime.now() - last_sync_time).seconds // 60
            status_text += f" (synced {ago} min ago)"

        cv2.putText(display_frame, status_text, (25, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        fps = 1 / (now - prev_time) if now > prev_time else 0
        prev_time = now
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (25, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 180), 3)

        cv2.putText(display_frame,
                    "Started",
                    (25, display_frame.shape[0]-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.82, (220,220,255), 2)

        cv2.imshow(WINDOW_NAME, display_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

def launch_kiosk():
    if not load_all_from_supabase():
        messagebox.showwarning("Cloud Warning", "Failed to load data from Supabase.")

    root = tk.Tk()
    root.title("Ontech Attendance Kiosk")
    root.geometry("800x750")  # Slightly larger for better spacing
    root.configure(bg="#0f172a")
    root.resizable(False, False)

    # Header
    header_frame = tk.Frame(root, bg="#0f172a")
    header_frame.pack(fill="x", pady=(40, 20))

    tk.Label(header_frame, text="Ontech Attendance",
             font=("Helvetica", 36, "bold"), fg="#f97316", bg="#0f172a").pack()

    sync_text = f"{datetime.date.today():%Y-%m-%d} • {len(face_db)} employees registered"
    if last_sync_time:
        ago = (datetime.datetime.now() - last_sync_time).seconds // 60
        sync_text += f" (cloud sync {ago} min ago)"

    tk.Label(header_frame, text=sync_text,
             font=("Helvetica", 16), fg="#94a3b8", bg="#0f172a").pack(pady=10)

    # Status box
    status_frame = tk.Frame(root, bg="#1e293b", bd=2, relief="flat")
    status_frame.pack(pady=30, padx=50, fill="x")

    tk.Label(status_frame, text="Ready to Scan",
             font=("Helvetica", 24, "bold"), fg="#e2e8f0", bg="#1e293b").pack(pady=20)

    tk.Label(status_frame, text="Position your face clearly in front of the camera",
             font=("Helvetica", 14), fg="#cbd5e1", bg="#1e293b", wraplength=600, justify="center").pack(pady=10)

    # Buttons - centered, even spacing
    btn_frame = tk.Frame(root, bg="#0f172a")
    btn_frame.pack(pady=40, padx=80, fill="x")

    button_style = {
        "font": ("Helvetica", 16, "bold"),
        "width": 30,
        "height": 2,
        "bd": 0,
        "relief": "flat",
        "cursor": "hand2",
        "activebackground": "#334155",
        "padx": 20,
        "pady": 10
    }

    def create_button(text, command, bg, fg="#ffffff"):
        btn = tk.Button(btn_frame, text=text, command=command, bg=bg, fg=fg, **button_style)
        btn.pack(pady=12, fill="x")
        btn.bind("<Enter>", lambda e: btn.config(bg="#334155"))
        btn.bind("<Leave>", lambda e: btn.config(bg=bg))
        return btn

    create_button(
        "Start Face Recognition",
        lambda: threading.Thread(target=run_attendance_recognition, daemon=True).start(),
        "#f97316", "#000000"
    )

    create_button("View All Employees", show_employee_list, "#22c55e")
    create_button("Today's Attendance", show_today_attendance, "#06b6d4")
    create_button("Sync from Cloud Now", 
                  lambda: load_all_from_supabase() or messagebox.showinfo("Sync", f"Reloaded {len(face_db)} employees"),
                  "#8b5cf6", "#ffffff")
    create_button("Exit", root.quit, "#ef4444")

    # Footer
    footer = tk.Label(root, text="Powered by InsightFace + MediaPipe + Supabase • Ontech",
                      font=("Helvetica", 10), fg="#475569", bg="#0f172a")
    footer.pack(side="bottom", pady=30)

    root.mainloop()

# ────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────
if __name__ == "__main__":
    if show_splash():
        launch_kiosk()