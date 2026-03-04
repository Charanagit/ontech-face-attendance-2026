import streamlit as st
import os
import io
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from insightface.app import FaceAnalysis
import datetime
import base64
from supabase import create_client, Client
from datetime import datetime
import pytz

# ────────────────────────────────────────────────
# Page config — MUST BE FIRST
# ────────────────────────────────────────────────
st.set_page_config(page_title="Ontech Employee Manager", layout="wide")

# Colombo time (UTC+5:30)
COLOMBO_TZ = pytz.timezone("Asia/Colombo")

def today_colombo():
    return datetime.now(COLOMBO_TZ).date()

def now_colombo():
    return datetime.now(COLOMBO_TZ)

# ────────────────────────────────────────────────
# Supabase client
# ────────────────────────────────────────────────
supabase: Client = create_client(
    st.secrets["SUPABASE_URL"],
    st.secrets["SUPABASE_KEY"]
)

# Connection check
try:
    supabase.table("employees").select("emp_code", count="planned").limit(0).execute()
    st.sidebar.success("Connected to Supabase ✓")
except Exception as e:
    st.sidebar.error(f"Supabase connection issue: {str(e)}")

# ────────────────────────────────────────────────
# Styling (same as before)
# ────────────────────────────────────────────────
OCEAN_BLUE   = "#0066cc"
DEEP_BLUE    = "#004080"
LIGHT_BLUE   = "#3A3433"
GREEN_ACCENT = "#2e7d32"
LIGHT_GREEN  = "#05580C"
PURPLE_ACCENT = "#7e57c2"
LIGHT_PURPLE = "#5A1919"
GRAY_BG      = "#0f1316"
TEXT_DARK    = "#1a1a2e"

st.markdown(f"""
    <style>
    .stApp {{ background-color: {GRAY_BG}; }}
    .block-container {{ padding-top: 2rem !important; padding-bottom: 2rem !important; }}
    h1, h2, h3 {{ color: {DEEP_BLUE}; }}
    .stButton > button[kind="primary"] {{
        background-color: {OCEAN_BLUE}; color: white; border: none; border-radius: 6px; padding: 0.6rem 1.2rem;
    }}
    .stButton > button[kind="primary"]:hover {{ background-color: {LIGHT_BLUE}; color: {OCEAN_BLUE}; }}
    hr {{ background-color: {OCEAN_BLUE}; height: 2px; border: none; }}
    .dataframe {{ background-color: white; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; }}
    </style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
# Paths (only for local photos - no pickle anymore)
# ────────────────────────────────────────────────
BASE_FOLDER = "data"
DATASET_FOLDER = os.path.join(BASE_FOLDER, "dataset")
os.makedirs(DATASET_FOLDER, exist_ok=True)

# ────────────────────────────────────────────────
# Session state
# ────────────────────────────────────────────────
if "selected_emp_code" not in st.session_state:
    st.session_state.selected_emp_code = None
if "last_processed" not in st.session_state:
    st.session_state.last_processed = None
if "save_result" not in st.session_state:
    st.session_state.save_result = None
if "save_messages" not in st.session_state:
    st.session_state.save_messages = []

# ────────────────────────────────────────────────
# Face model (cached)
# ────────────────────────────────────────────────
@st.cache_resource
def get_face_model():
    with st.spinner("Loading InsightFace buffalo_s model..."):
        app = FaceAnalysis(name="buffalo_s")
        app.prepare(ctx_id=0, det_size=(640, 640))
    st.success("Model loaded", icon="✅")
    return app

app = get_face_model()

# ────────────────────────────────────────────────
# Utils
# ────────────────────────────────────────────────
def load_image(file_bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img = img.convert("RGB")
        return np.array(img), img
    except Exception as e:
        st.warning(f"Failed to load image: {e}")
        return None, None

def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec

def has_embedding(emp_code: str) -> str:
    try:
        response = supabase.table("face_embeddings").select("embedding_base64").eq("emp_code", emp_code).execute()
        if response.data and response.data[0].get("embedding_base64"):
            b64 = response.data[0]["embedding_base64"]
            return f"Yes ({len(b64)} chars)"
        return "No"
    except Exception as e:
        st.warning(f"Embedding check failed for {emp_code}: {str(e)}")
        return "Error"

# ────────────────────────────────────────────────
# Core: process photos → mean embedding → save to Supabase
# ────────────────────────────────────────────────
def process_employee(emp_code, full_name, department, designation, mobile, notes, uploaded_files):
    messages = []
    emp_code = emp_code.strip().upper()

    if not emp_code:
        messages.append("Employee Code is required")
        return messages

    embeddings = []
    emp_folder = os.path.join(DATASET_FOLDER, emp_code)
    os.makedirs(emp_folder, exist_ok=True)

    if uploaded_files:
        with st.spinner("Processing up to 3 valid face photos..."):
            valid_count = 0
            for up_file in uploaded_files:
                if valid_count >= 3:
                    messages.append(f"Stopped at 3 valid photos — {up_file.name} ignored")
                    break

                try:
                    if up_file.size > 5 * 1024 * 1024:
                        messages.append(f"{up_file.name} too large (>5MB) → skipped")
                        continue

                    img_cv, img_pil = load_image(up_file.getvalue())
                    if img_cv is None:
                        continue

                    faces = app.get(img_cv)
                    if len(faces) != 1:
                        messages.append(f"{up_file.name}: {len(faces)} faces → skipped")
                        continue

                    face = faces[0]
                    if face.det_score < 0.75:
                        messages.append(f"{up_file.name}: low confidence ({face.det_score:.2f}) → skipped")
                        continue

                    emb = normalize(face.embedding)
                    embeddings.append(emb)
                    valid_count += 1

                    fname = os.path.splitext(up_file.name)[0] + ".png"
                    img_pil.save(os.path.join(emp_folder, fname))
                    messages.append(f"{up_file.name}: valid (conf {face.det_score:.2f})")

                except Exception as e:
                    messages.append(f"Error processing {up_file.name}: {str(e)}")

    mean_emb_norm = None
    if len(embeddings) >= 1:  # lowered to 1 for testing — you can raise to 3 later
        mean_emb = np.mean(embeddings, axis=0)
        mean_emb_norm = normalize(mean_emb)
        messages.append(f"Mean embedding created from {len(embeddings)} photos")
    else:
        messages.append("No valid photos → saving metadata only (no embedding)")

    # Save / upsert to Supabase
    with st.spinner("Saving to Supabase..."):
        try:
            # Employee metadata
            emp_data = {
                "emp_code": emp_code,
                "full_name": full_name.strip() or None,
                "department": department.strip() or None,
                "designation": designation.strip() or None,
                "mobile": mobile.strip() or None,
                "notes": notes.strip() or None,
            }
            supabase.table("employees").upsert(emp_data, on_conflict="emp_code").execute()

            # Embedding (if we have one)
            if mean_emb_norm is not None:
                b64_encoded = base64.b64encode(mean_emb_norm.tobytes()).decode('utf-8')
                emb_data = {
                    "emp_code": emp_code,
                    "embedding_base64": b64_encoded
                }
                supabase.table("face_embeddings").upsert(emb_data, on_conflict="emp_code").execute()
                messages.append(f"Embedding saved ({len(b64_encoded)} chars base64)")
            else:
                messages.append("No embedding saved (metadata only)")

        except Exception as e:
            messages.append(f"Supabase save error: {str(e)}")

    return messages

# ────────────────────────────────────────────────
# Attendance functions
# ────────────────────────────────────────────────
def get_today_attendance():
    today = today_colombo().isoformat()
    try:
        response = supabase.table("attendance")\
            .select("emp_code, checkin_time, checkout_time")\
            .eq("checkin_date", today)\
            .order("checkin_time", desc=True)\
            .execute()

        records = []
        for r in response.data:
            code = r["emp_code"]
            emp_resp = supabase.table("employees").select("full_name, department").eq("emp_code", code).execute()
            name = emp_resp.data[0]["full_name"] if emp_resp.data else code
            dept = emp_resp.data[0]["department"] if emp_resp.data else ""
            records.append({
                "emp_code": code,
                "name": name,
                "department": dept,
                "checkin_time": r["checkin_time"] or "-",
                "checkout_time": r["checkout_time"] or "-"
            })
        return records
    except Exception as e:
        st.error(f"Failed to load today's attendance: {e}")
        return []

# ────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────
st.title("🧑‍💼 Ontech Employee & Attendance Manager")
st.markdown(f"<h3 style='color:{PURPLE_ACCENT};'>Admin Control Panel</h3>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Select Section",
    ["Main Dashboard", "Register / Edit Employee", "Today's Attendance"]
)

# ────────────────────────────────────────────────
# Main Dashboard
# ────────────────────────────────────────────────
if page == "Main Dashboard":
    st.subheader("Registered Employees & Today's Attendance")

    try:
        emp_response = supabase.table("employees").select("*").execute()
        employees = emp_response.data or []

        emb_response = supabase.table("face_embeddings").select("emp_code").execute()
        has_emb = {row["emp_code"]: True for row in emb_response.data or []}

        today = today_colombo().isoformat()
        att_response = supabase.table("attendance").select("*").eq("checkin_date", today).execute()

        present_dict = {}
        for r in att_response.data or []:
            code = r["emp_code"]
            status = f"Yes ({r['checkin_time']})" if r["checkin_time"] else "No"
            if r.get("checkout_time"):
                status += f" - Out ({r['checkout_time']})"
            present_dict[code] = status

        present_count = len(present_dict)

        if employees:
            st.caption(f"**Today ({today}):** {present_count} / {len(employees)} checked in")

            data = []
            for emp in employees:
                code = emp["emp_code"]
                name = emp.get("full_name") or code
                dept = emp.get("department") or ""
                desig = emp.get("designation") or ""
                mob = emp.get("mobile") or ""
                notes = emp.get("notes") or ""

                data.append({
                    "Code": code,
                    "Name": name,
                    "Department": dept,
                    "Designation": desig,
                    "Mobile": mob,
                    "Notes": notes[:100] + "…" if len(notes or "") > 100 else notes,
                    "Has Embedding": "Yes" if code in has_emb else "No",
                    "Present Today": present_dict.get(code, "No")
                })

            df = pd.DataFrame(data)

            def highlight_present(row):
                color = LIGHT_GREEN if "Yes" in row["Present Today"] else "#f8d7da"
                return [f'background-color: {color}' if col == "Present Today" else '' for col in df.columns]

            st.dataframe(
                df.style.apply(highlight_present, axis=1),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No employees registered yet.")
    except Exception as e:
        st.error(f"Dashboard load failed: {e}")

# ────────────────────────────────────────────────
# Register / Edit Employee
# ────────────────────────────────────────────────
elif page == "Register / Edit Employee":
    st.subheader("Add / Edit Employee")

    emp_code_value = full_name_value = department_value = designation_value = mobile_value = notes_value = ""

    if st.session_state.selected_emp_code:
        code = st.session_state.selected_emp_code
        try:
            resp = supabase.table("employees").select("*").eq("emp_code", code).execute()
            if resp.data:
                emp = resp.data[0]
                full_name_value = emp.get("full_name", "")
                department_value = emp.get("department", "")
                designation_value = emp.get("designation", "")
                mobile_value = emp.get("mobile", "")
                notes_value = emp.get("notes", "")
                emp_code_value = code
                st.info(f"Editing: **{code}** (add photos to update embedding)")
        except Exception as e:
            st.error(f"Failed to load employee: {e}")

        if st.button("× Cancel editing"):
            st.session_state.selected_emp_code = None
            st.rerun()

    col1, col2 = st.columns([3, 3])

    with col1:
        emp_code_input = st.text_input("Employee Code (unique)", value=emp_code_value, disabled=bool(st.session_state.selected_emp_code))
        full_name = st.text_input("Full Name", value=full_name_value)
        department = st.text_input("Department / Team", value=department_value)

    with col2:
        designation = st.text_input("Designation / Role", value=designation_value)
        mobile = st.text_input("Mobile Number", value=mobile_value)

    notes = st.text_area("Notes / Remarks", value=notes_value, height=110)

    st.subheader("Face Photos (up to 3 processed)")
    uploaded_files = st.file_uploader("Upload clear face photos", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

    if uploaded_files:
        cols = st.columns(min(5, len(uploaded_files)))
        for i, file in enumerate(uploaded_files):
            try:
                img = Image.open(file)
                cols[i % len(cols)].image(img, caption=file.name, use_column_width=True)
            except:
                pass

    col_btn1, col_btn2 = st.columns([4, 2])

    with col_btn1:
        if st.button("💾 Save / Update Employee", type="primary", use_container_width=True):
            emp_code = st.session_state.selected_emp_code or emp_code_input.strip().upper()

            if not emp_code:
                st.error("Employee Code is required")
            else:
                with st.spinner("Processing..."):
                    msgs = process_employee(
                        emp_code=emp_code,
                        full_name=full_name,
                        department=department,
                        designation=designation,
                        mobile=mobile,
                        notes=notes,
                        uploaded_files=uploaded_files
                    )
                    for m in msgs:
                        if "error" in m.lower():
                            st.error(m)
                        else:
                            st.info(m)

                    st.session_state.last_processed = emp_code
                st.rerun()

    with col_btn2:
        if st.button("Clear Form", use_container_width=True):
            for k in ["emp_code_input", "full_name", "department", "designation", "mobile", "notes"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.session_state.selected_emp_code = None
            st.rerun()

# ────────────────────────────────────────────────
# Today's Attendance
# ────────────────────────────────────────────────
elif page == "Today's Attendance":
    st.subheader("Today's Attendance")

    records = get_today_attendance()

    if records:
        df = pd.DataFrame(records)
        df["Status"] = df["checkout_time"].apply(lambda x: "Checked Out" if x != "-" else "Present")
        st.dataframe(df[["emp_code", "name", "department", "checkin_time", "checkout_time", "Status"]], use_container_width=True)
        st.success(f"Total present today: {len(records)}")
    else:
        st.info("No attendance records today yet.")