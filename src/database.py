from pathlib import Path
import sqlite3

BASE_DIR = Path(__file__).resolve().parent

DB_DIR = BASE_DIR
DB_DIR.mkdir(exist_ok=True)

DB_PATH = DB_DIR / "database.db"

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id TEXT,
    timestamp TEXT,
    loan_status TEXT
)
""")

conn.commit()