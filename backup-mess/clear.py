# clear_checkins.py
# Run this script to reset today's attendance (or all attendance) for testing

import sqlite3
import datetime
import os

# Same paths as your other scripts
BASE_FOLDER = "data"
DB_PATH = os.path.join(BASE_FOLDER, "employees.db")

def clear_todays_checkins():
    today = datetime.date.today().isoformat()
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            
            # Show how many records will be deleted
            c.execute("SELECT COUNT(*) FROM attendance WHERE checkin_date = ?", (today,))
            count = c.fetchone()[0]
            print(f"Found {count} check-in records for today ({today})")
            
            if count == 0:
                print("Nothing to clear today.")
                return
            
            confirm = input("Delete ALL today's check-ins? (y/n): ").strip().lower()
            if confirm != 'y':
                print("Cancelled.")
                return
            
            # Delete today's records
            c.execute("DELETE FROM attendance WHERE checkin_date = ?", (today,))
            conn.commit()
            
            print(f"Successfully deleted {count} check-in records for {today}.")
            
    except sqlite3.Error as e:
        print(f"Database error: {e}")

def clear_all_checkins():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            
            c.execute("SELECT COUNT(*) FROM attendance")
            count = c.fetchone()[0]
            print(f"Found {count} check-in records in total")
            
            if count == 0:
                print("No check-ins to clear.")
                return
            
            confirm = input("Delete ALL check-ins EVER recorded? This cannot be undone! (y/n): ").strip().lower()
            if confirm != 'y':
                print("Cancelled.")
                return
            
            c.execute("DELETE FROM attendance")
            conn.commit()
            
            print(f"Successfully deleted ALL {count} check-in records.")
            
    except sqlite3.Error as e:
        print(f"Database error: {e}")

def main():
    print("=== Attendance Clear Tool (for testing) ===")
    print("1. Clear only today's check-ins")
    print("2. Clear ALL check-ins ever recorded")
    print("3. Exit")
    
    choice = input("\nChoose (1/2/3): ").strip()
    
    if choice == '1':
        clear_todays_checkins()
    elif choice == '2':
        clear_all_checkins()
    elif choice == '3':
        print("Goodbye.")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        print(f"Database not found: {DB_PATH}")
    else:
        main()