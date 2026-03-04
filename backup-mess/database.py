# database.py
import sqlite3
import os
import sys
import datetime
import numpy as np
from typing import Tuple, List, Optional, Dict

# Consistent path (works with PyInstaller .exe)
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_FOLDER = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(BASE_FOLDER, "employees.db")


def normalize_emp_code(code: str) -> str:
    """Clean emp_code — always uppercase for reliable matching."""
    return str(code).strip().upper() if code else ""


def init_db() -> None:
    """Create database and tables if they don't exist."""
    os.makedirs(BASE_FOLDER, exist_ok=True)
    
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        
        # Employees table with embedding as BLOB
        c.execute('''
            CREATE TABLE IF NOT EXISTS employees (
                emp_code        TEXT PRIMARY KEY,
                full_name       TEXT,
                department      TEXT,
                designation     TEXT,
                mobile          TEXT,
                registered_date TEXT,
                notes           TEXT,
                embedding       BLOB
            )
        ''')
        
        # Attendance table
        c.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                emp_code        TEXT NOT NULL,
                checkin_date    TEXT NOT NULL,
                checkin_time    TEXT NOT NULL,
                checkout_time   TEXT,
                FOREIGN KEY (emp_code) REFERENCES employees(emp_code)
            )
        ''')
        
        # Add index for faster lookups (very helpful when attendance grows)
        try:
            c.execute("CREATE INDEX IF NOT EXISTS idx_attendance_emp_date ON attendance(emp_code, checkin_date)")
        except sqlite3.OperationalError:
            pass
        
        # Migration: add checkout_time if missing
        try:
            c.execute("ALTER TABLE attendance ADD COLUMN checkout_time TEXT")
            print("Added checkout_time column (if missing)")
        except sqlite3.OperationalError:
            pass
        
        conn.commit()


def save_employee(
    emp_code: str,
    full_name: str = "",
    department: str = "",
    designation: str = "",
    mobile: str = "",
    notes: str = "",
    embedding: Optional[np.ndarray] = None
) -> bool:
    init_db()
    emp_code = normalize_emp_code(emp_code)
    if not emp_code:
        print("Error: emp_code cannot be empty")
        return False
    
    registered_date = datetime.datetime.now().isoformat()  # ISO for consistency
    embedding_bytes = embedding.tobytes() if embedding is not None else None
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute('''
                INSERT OR REPLACE INTO employees 
                (emp_code, full_name, department, designation, mobile, registered_date, notes, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                emp_code,
                full_name.strip(),
                department.strip(),
                designation.strip(),
                mobile.strip(),
                registered_date,
                notes.strip(),
                embedding_bytes
            ))
            conn.commit()
            print(f"Employee saved/updated: {emp_code} (embedding: {'yes' if embedding is not None else 'no'})")
            return True
    except sqlite3.Error as e:
        print(f"Database error saving employee {emp_code}: {e}")
        return False


def load_employee_info(emp_code: str) -> Tuple[str, str, str, str, str]:
    init_db()
    emp_code = normalize_emp_code(emp_code)
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute('''
                SELECT full_name, department, designation, mobile, notes 
                FROM employees 
                WHERE emp_code = ?
            ''', (emp_code,))
            row = c.fetchone()
            if row:
                full_name, dept, desig, mob, nt = row
                return (
                    full_name or emp_code,
                    dept or "",
                    desig or "",
                    mob or "",
                    nt or ""
                )
            return emp_code, "", "", "", ""
    except sqlite3.Error as e:
        print(f"Error loading employee {emp_code}: {e}")
        return emp_code, "", "", "", ""


def load_all_embeddings() -> Dict[str, np.ndarray]:
    """Load all employee codes → embeddings for kiosk face recognition."""
    init_db()
    face_db = {}
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute('''
                SELECT emp_code, embedding 
                FROM employees 
                WHERE embedding IS NOT NULL
            ''')
            rows = c.fetchall()
            for code, emb_bytes in rows:
                if emb_bytes:
                    try:
                        emb_array = np.frombuffer(emb_bytes, dtype=np.float32)
                        # Optional safety: InsightFace usually gives 512-dim vectors
                        if emb_array.size == 512:
                            face_db[code] = emb_array
                        else:
                            print(f"Invalid embedding size {emb_array.size} for {code} — skipping")
                    except ValueError as ve:
                        print(f"Buffer error for {code}: {ve} — skipping")
            print(f"Loaded {len(face_db)} valid embeddings from database")
            return face_db
    except sqlite3.Error as e:
        print(f"Error loading embeddings: {e}")
        return {}


def get_all_employees() -> List[Tuple[str, str, str, str, str, str]]:
    """Return list of all registered employees: (code, name, dept, desig, mobile, notes)"""
    init_db()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute('''
                SELECT emp_code, full_name, department, designation, mobile, notes 
                FROM employees 
                ORDER BY emp_code
            ''')
            rows = c.fetchall()
            print(f"Total employees in DB: {len(rows)}")
            return rows
    except sqlite3.Error as e:
        print(f"Error fetching employees: {e}")
        return []


def load_all_employees() -> List[Tuple[str, str, str, str, str, str]]:
    """
    Alias for get_all_employees() — fixes yellow underline in recognize_webcam.py
    """
    return get_all_employees()


def mark_present(emp_code: str) -> bool:
    emp_code = normalize_emp_code(emp_code)
    today = datetime.date.today().isoformat()
    now_time = datetime.datetime.now().strftime("%H:%M:%S")
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute('''
                SELECT 1 FROM attendance 
                WHERE emp_code = ? AND checkin_date = ?
            ''', (emp_code, today))
            
            if c.fetchone():
                print(f"{emp_code} already checked in today")
                return False
            
            c.execute('''
                INSERT INTO attendance (emp_code, checkin_date, checkin_time)
                VALUES (?, ?, ?)
            ''', (emp_code, today, now_time))
            conn.commit()
            print(f"Check-in recorded: {emp_code} at {now_time}")
            return True
    except sqlite3.Error as e:
        print(f"Check-in error: {e}")
        return False


def mark_out(emp_code: str) -> bool:
    emp_code = normalize_emp_code(emp_code)
    today = datetime.date.today().isoformat()
    now_time = datetime.datetime.now().strftime("%H:%M:%S")
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            # Get the most recent record without checkout
            c.execute("""
                SELECT id, checkin_time, checkout_time 
                FROM attendance 
                WHERE emp_code = ? AND checkin_date = ?
                ORDER BY id DESC LIMIT 1
            """, (emp_code, today))
            
            row = c.fetchone()
            if not row or row[1] is None:
                print(f"No check-in found for {emp_code} today")
                return False
            
            if row[2] is not None:
                print(f"{emp_code} already checked out")
                return False
            
            c.execute("""
                UPDATE attendance 
                SET checkout_time = ? 
                WHERE id = ?
            """, (now_time, row[0]))
            
            conn.commit()
            print(f"Check-out recorded: {emp_code} at {now_time}")
            return True
    except sqlite3.Error as e:
        print(f"Check-out error: {e}")
        return False


def is_present_today(emp_code: str) -> str:
    emp_code = normalize_emp_code(emp_code)
    today = datetime.date.today().isoformat()
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT checkin_time, checkout_time FROM attendance 
                WHERE emp_code = ? AND checkin_date = ?
                ORDER BY checkin_time DESC LIMIT 1
            """, (emp_code, today))
            row = c.fetchone()
            if row:
                checkin, checkout = row
                if checkout:
                    return f"Checked out at {checkout}"
                return f"Yes ({checkin})"
            return "No"
    except sqlite3.Error as e:
        print(f"Presence check error: {e}")
        return "Error"


def get_attendance_for_employee_date(emp_code: str, date: str) -> Dict[str, str]:
    emp_code = normalize_emp_code(emp_code)
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT checkin_time, checkout_time 
                FROM attendance 
                WHERE emp_code = ? AND checkin_date = ?
                ORDER BY id DESC LIMIT 1
            """, (emp_code, date))
            row = c.fetchone()
            if row:
                checkin, checkout = row
                return {
                    "status": "present",
                    "checkin_time": checkin,
                    "checkout_time": checkout or "Not checked out"
                }
            return {"status": "absent"}
    except sqlite3.Error as e:
        print(f"Error: {e}")
        return {"status": "error"}


def get_attendance_history_for_employee(emp_code: str, limit: int = 30) -> List[Dict]:
    emp_code = normalize_emp_code(emp_code)
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT checkin_date, checkin_time, checkout_time
                FROM attendance
                WHERE emp_code = ?
                ORDER BY checkin_date DESC
                LIMIT ?
            """, (emp_code, limit))
            rows = c.fetchall()
            return [
                {
                    "date": date,
                    "checkin_time": cin,
                    "checkout_time": cout or "-",
                    "status": "Checked out" if cout else "Present"
                } for date, cin, cout in rows
            ]
    except sqlite3.Error as e:
        print(f"History error: {e}")
        return []


def get_attendance_for_date(date: str) -> List[Dict]:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT 
                    e.emp_code, e.full_name, e.department,
                    a.checkin_time, a.checkout_time
                FROM attendance a
                JOIN employees e ON a.emp_code = e.emp_code
                WHERE a.checkin_date = ?
                ORDER BY a.checkin_time DESC
            """, (date,))
            rows = c.fetchall()
            return [
                {
                    "emp_code": code,
                    "name": name,
                    "department": dept,
                    "checkin_time": cin,
                    "checkout_time": cout or "-"
                } for code, name, dept, cin, cout in rows
            ]
    except sqlite3.Error as e:
        print(f"Daily attendance error: {e}")
        return []


def get_today_present() -> List[Dict]:
    today = datetime.date.today().isoformat()
    return get_attendance_for_date(today)


# ── Debug helper ──────────────────────────────────────────────
def debug_db_status():
    print("\n=== Database Status Debug ====")
    employees = get_all_employees()
    print(f"Employees registered: {len(employees)}")
    if employees:
        print("Sample:", employees[:3])
    
    embeddings = load_all_embeddings()
    print(f"Embeddings loaded: {len(embeddings)}")
    
    today = datetime.date.today().isoformat()
    print(f"Today's attendance records: {len(get_today_present())}")


if __name__ == "__main__":
    debug_db_status()