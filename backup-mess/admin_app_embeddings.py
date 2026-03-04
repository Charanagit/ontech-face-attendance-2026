import streamlit as st
import os
import io
import numpy as np
import pandas as pd
from PIL import Image
import datetime
import base64  # For embedding serialization
from supabase import create_client, Client
from datetime import datetime
import pytz

# ────────────────────────────────────────────────
# Page config — MUST BE FIRST
# ────────────────────────────────────────────────
st.set_page_config(page_title="Ontech Employee Manager", layout="wide")

# Force Colombo time (UTC+5:30)
COLOMBO_TZ = pytz.timezone("Asia/Colombo")

def today_colombo():
    return datetime.now(COLOMBO_TZ).date()

def now_colombo():
    return datetime.now(COLOMBO_TZ)

# ────────────────────────────────────────────────
# Supabase client — works locally AND on cloud
# ────────────────────────────────────────────────
# Supabase client — works locally AND on cloud
SUPABASE_URL = os.getenv("SUPABASE_URL") or "https://crujjurupavknjwdjjmj.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNydWpqdXJ1cGF2a25qd2Rqam1qIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzA5NjI0MTAsImV4cCI6MjA4NjUzODQxMH0.MdQDrEHOyQ0mI6HGX986lNMw5cpj5pfUCnKFh88pnzw"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Connection check with spinner
with st.spinner("Connecting to Supabase..."):
    try:
        supabase.table("employees").select("emp_code", count="planned").limit(0).execute()
        st.sidebar.success("Connected to Supabase ✓")
    except Exception as e:
        st.sidebar.error(f"Supabase connection failed: {str(e)}")
        st.sidebar.info("Check URL/key or Supabase status")

# ────────────────────────────────────────────────
# Styling
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

def has_embedding(emp_code):
    try:
        response = supabase.table("face_embeddings").select("embedding_base64").eq("emp_code", emp_code).execute()
        if response.data:
            b64_str = response.data[0].get("embedding_base64")
            if b64_str:
                return f"Yes ({len(b64_str)} chars)"
        return "No"
    except Exception as e:
        st.warning(f"Embedding check failed for {emp_code}: {str(e)}")
        return "Error"

# ────────────────────────────────────────────────
# Core: process photos → embedding → save to Supabase
# ────────────────────────────────────────────────
def process_employee(emp_code, full_name, department, designation, mobile, notes, uploaded_files):
    messages = []

    emp_code = emp_code.strip().upper()
    if not emp_code:
        st.error("Employee Code is required")
        return messages

    embedding_to_save = None

    if uploaded_files:
        embeddings = []

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
                        messages.append(f"{up_file.name} load failed → skipped")
                        continue

                    # Dashboard mode: simulate processing (real detection in kiosk)
                    messages.append(f"{up_file.name}: photo loaded (embedding handled in kiosk app)")

                except Exception as e:
                    messages.append(f"Error processing {up_file.name}: {str(e)} → skipped")

        if len(uploaded_files) > 0:
            messages.append("Embedding simulation complete (real processing in kiosk app)")

    # Save metadata to Supabase
    with st.spinner("Saving to Supabase..."):
        try:
            emp_data = {
                "emp_code": emp_code,
                "full_name": full_name.strip() or None,
                "department": department.strip() or None,
                "designation": designation.strip() or None,
                "mobile": mobile.strip() or None,
                "notes": notes.strip() or None,
            }

            emp_response = supabase.table("employees").upsert(
                emp_data, on_conflict="emp_code"
            ).execute()

            messages.append(f"Metadata saved ({emp_response.count} row(s))")

            # Embedding save skipped here — handled in kiosk app
            messages.append("Embedding save skipped (handled in kiosk app)")

            return messages

        except Exception as e:
            error_str = str(e)
            messages.append(f"Supabase error: {error_str}")
            st.error(f"Save failed: {error_str}")
            import traceback
            traceback.print_exc()
            return messages

# ────────────────────────────────────────────────
# Main UI & Navigation
# ────────────────────────────────────────────────
st.title("🧑‍💼 Ontech Employee & Attendance Manager")
st.markdown(f"<h3 style='color:{PURPLE_ACCENT};'>Admin Control Panel</h3>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Select Section",
    ["Main Dashboard (Overview)", "Register / Edit Employee", "Today's Attendance", "Employee Attendance History", "Daily Attendance Report"]
)

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

def get_employee_history(emp_code):
    try:
        response = supabase.table("attendance")\
            .select("checkin_date, checkin_time, checkout_time")\
            .eq("emp_code", emp_code)\
            .order("checkin_date", desc=True)\
            .execute()
        
        history = []
        for r in response.data:
            status = "Checked Out" if r["checkout_time"] else "Present"
            history.append({
                "date": r["checkin_date"],
                "checkin_time": r["checkin_time"] or "-",
                "checkout_time": r["checkout_time"] or "-",
                "status": status
            })
        return history
    except Exception as e:
        st.error(f"Failed to load history for {emp_code}: {e}")
        return []

# ────────────────────────────────────────────────
# Main Dashboard
# ────────────────────────────────────────────────
if page == "Main Dashboard (Overview)":
    st.subheader("Registered Employees & Today's Attendance Status")

    try:
        emp_response = supabase.table("employees").select(
            "emp_code, full_name, department, designation, mobile, notes"
        ).execute()
        employees = emp_response.data or []

        emb_response = supabase.table("face_embeddings").select("emp_code").execute()
        has_emb = {row["emp_code"]: True for row in emb_response.data or []}

        today = today_colombo().isoformat()
        att_response = supabase.table("attendance").select(
            "emp_code, checkin_time, checkout_time"
        ).eq("checkin_date", today).execute()

        present_dict = {}
        for r in att_response.data or []:
            code = r["emp_code"]
            status = f"Yes ({r['checkin_time']})" if r["checkin_time"] else "No"
            if r["checkout_time"]:
                status += f" - Out ({r['checkout_time']})"
            present_dict[code] = status

        present_count = len(present_dict)

    except Exception as e:
        st.error(f"Failed to load data: {e}")
        employees = []
        present_dict = {}
        present_count = 0

    if employees:
        st.caption(f"**Today ({today_colombo():%Y-%m-%d})**: {present_count} / {len(employees)} checked in")

        employees_data = []
        for emp in employees:
            code = emp["emp_code"]
            name = emp.get("full_name") or code
            dept = emp.get("department") or ""
            desig = emp.get("designation") or ""
            mob = emp.get("mobile") or ""
            notes = emp.get("notes") or ""

            emb_status = "Yes" if code in has_emb else "No"
            present_status = present_dict.get(code, "No")

            employees_data.append({
                "Code": code,
                "Name": name,
                "Department": dept,
                "Designation": desig,
                "Mobile": mob,
                "Notes": notes[:100] + "…" if len(notes) > 100 else notes,
                "Has Embedding": emb_status,
                "Present Today": present_status
            })

        df = pd.DataFrame(employees_data)

        def highlight_present(row):
            color = LIGHT_GREEN if "Yes" in row["Present Today"] else "#f8d7da"
            return [f'background-color: {color}' if col == "Present Today" else '' for col in df.columns]

        styled_df = df.style.apply(highlight_present, axis=1)

        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Notes": st.column_config.TextColumn(width="medium"),
                "Has Embedding": st.column_config.TextColumn(width="small"),
                "Present Today": st.column_config.TextColumn(width="medium")
            }
        )

        if st.button("🔄 Refresh Dashboard", type="primary"):
            st.rerun()

        st.markdown("**Quick Edit Employees:**")
        cols = st.columns(6)
        for i, emp in enumerate(employees_data):
            with cols[i % 6]:
                label = f"✏ {emp['Code']}"
                if emp['Name'] != emp['Code']:
                    label += f" – {emp['Name'][:15]}…"
                if st.button(label, key=f"edit_{emp['Code']}", use_container_width=True):
                    st.session_state.selected_emp_code = emp["Code"]
                    st.rerun()
    else:
        st.info("No employees registered yet. Add someone below.", icon="ℹ️")

# ────────────────────────────────────────────────
# Register / Edit Employee
# ────────────────────────────────────────────────
elif page == "Register / Edit Employee":
    st.subheader("Add / Edit Employee")

    emp_code_value = full_name_value = department_value = designation_value = mobile_value = notes_value = ""

    editing = bool(st.session_state.selected_emp_code)

    if editing:
        code = st.session_state.selected_emp_code
        try:
            resp = supabase.table("employees").select("*").eq("emp_code", code).execute()
            if resp.data:
                emp = resp.data[0]
                full_name_value = emp.get("full_name", "") or ""
                department_value = emp.get("department", "") or ""
                designation_value = emp.get("designation", "") or ""
                mobile_value = emp.get("mobile", "") or ""
                notes_value = emp.get("notes", "") or ""
                emp_code_value = code
                st.info(f"Editing employee: **{code}** (add photos to update embedding)", icon="✏️")
        except Exception as e:
            st.error(f"Failed to load employee: {e}")

        if st.button("× Cancel editing", type="secondary"):
            st.session_state.selected_emp_code = None
            st.rerun()

    col1, col2 = st.columns([3, 3])

    with col1:
        emp_code_input = st.text_input("Employee Code (unique)", value=emp_code_value,
                                       key="emp_code_input", disabled=editing)
        full_name = st.text_input("Full Name", value=full_name_value, key="full_name")
        department = st.text_input("Department / Team", value=department_value, key="department")

    with col2:
        designation = st.text_input("Designation / Role", value=designation_value, key="designation")
        mobile = st.text_input("Mobile Number", value=mobile_value, key="mobile")

    notes = st.text_area("Notes / Remarks", value=notes_value, height=110, key="notes")

    st.subheader("Face Photos (up to 3 — embedding saved to Supabase)")
    uploaded_files = st.file_uploader(
        "Upload clear face photos (JPG/PNG only)",
        accept_multiple_files=True,
        type=["jpg", "jpeg", "png"],
        key="uploader"
    )

    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} file(s):")
        cols = st.columns(min(5, len(uploaded_files) or 1))
        for i, file in enumerate(uploaded_files):
            try:
                img = Image.open(file)
                cols[i % len(cols)].image(img, caption=file.name, use_container_width=True)
            except:
                st.warning(f"Cannot preview {file.name}")

    col_btn1, col_btn2 = st.columns([4, 2])

    with col_btn1:
        if st.button("💾 Save / Update Employee", type="primary", use_container_width=True):
            emp_code = (
                st.session_state.selected_emp_code
                if st.session_state.selected_emp_code
                else emp_code_input.strip().upper()
            )

            if not emp_code:
                st.error("Employee Code is required")
            else:
                st.session_state.save_result = None
                st.session_state.save_messages = []

                with st.spinner("Processing and saving..."):
                    msgs = process_employee(
                        emp_code=emp_code,
                        full_name=full_name,
                        department=department,
                        designation=designation,
                        mobile=mobile,
                        notes=notes,
                        uploaded_files=uploaded_files
                    )
                    st.session_state.save_messages = msgs

                has_error = any("error" in m.lower() or "fail" in m.lower() for m in msgs)
                if has_error:
                    st.session_state.save_result = "error"
                    st.error("Save had issues — read messages below")
                else:
                    st.session_state.save_result = "success"
                    st.success(f"**{emp_code}** saved/updated! Refresh dashboard to see changes.")
                    st.session_state.last_processed = emp_code

                for m in st.session_state.save_messages:
                    if "error" in m.lower() or "fail" in m.lower():
                        st.error(m)
                    else:
                        st.info(m)

    with col_btn2:
        if st.button("Clear Form", use_container_width=True):
            keys = ["emp_code_input", "full_name", "department", "designation", "mobile", "notes", "uploader"]
            for k in keys:
                if k in st.session_state:
                    del st.session_state[k]
            st.session_state.selected_emp_code = None
            st.session_state.last_processed = None
            st.session_state.save_result = None
            st.session_state.save_messages = []
            st.rerun()

# ────────────────────────────────────────────────
# Placeholder Pages
# ────────────────────────────────────────────────
elif page == "Today's Attendance":
    st.subheader("Today's Attendance")
    
    records = get_today_attendance()
    
    if records:
        df = pd.DataFrame(records)
        df["Status"] = df["checkout_time"].apply(lambda x: "Checked Out" if x != "-" else "Present")
        
        st.dataframe(
            df[["emp_code", "name", "department", "checkin_time", "checkout_time", "Status"]],
            use_container_width=True,
            hide_index=True
        )
        st.success(f"Total present today: {len(records)} employees")
    else:
        st.info("No attendance records today yet.", icon="ℹ️")

    if st.button("🔄 Refresh Today's Attendance"):
        st.rerun()

elif page == "Employee Attendance History":
    st.subheader("Employee Attendance History")

    try:
        emp_list = supabase.table("employees").select("emp_code, full_name").execute().data
        options = [f"{e['emp_code']} - {e['full_name'] or e['emp_code']}" for e in emp_list]
        emp_dict = {opt: opt.split(" - ")[0] for opt in options}
    except Exception as e:
        st.error(f"Failed to load employees: {e}")
        options = []

    selected = st.selectbox("Select Employee", [""] + options)

    if selected and selected != "":
        code = emp_dict[selected]
        history = get_employee_history(code)

        if history:
            df_hist = pd.DataFrame(history)
            st.dataframe(
                df_hist[["date", "checkin_time", "checkout_time", "status"]],
                use_container_width=True,
                hide_index=True
            )
            st.success(f"Found {len(history)} records for {selected}")
        else:
            st.info("No attendance history found for this employee.")

        if st.button("🔄 Refresh History"):
            st.rerun()
    else:
        st.info("Select an employee to view their attendance history.")

elif page == "Daily Attendance Report":
    st.subheader("Daily Attendance Report")
    st.info("Coming soon.")