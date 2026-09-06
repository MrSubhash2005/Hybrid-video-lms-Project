import sqlite3
from pathlib import Path
from src.config import STORAGE_DIR

DB_PATH = STORAGE_DIR.parent / "jobs.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                progress REAL NOT NULL DEFAULT 0,
                estimated_time_remaining REAL,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                error_message TEXT,
                output_path TEXT,
                output_url TEXT,
                voice TEXT,
                avatar TEXT,
                gender TEXT
            )
        """)


def create_job(
    job_id: str,
    created_at: str,
    voice: str = None,
    avatar: str = None,
    gender: str = None,
):
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO jobs (job_id, status, progress, estimated_time_remaining, created_at, voice, avatar, gender)
            VALUES (?, 'queued', 0.0, 30.0, ?, ?, ?, ?)
        """,
            (job_id, created_at, voice, avatar, gender),
        )


def get_job(job_id: str):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None


def update_job(job_id: str, **fields):
    if not fields:
        return

    set_clause = ", ".join(f"{k} = ?" for k in fields.keys())
    values = list(fields.values()) + [job_id]

    with get_connection() as conn:
        conn.execute(f"UPDATE jobs SET {set_clause} WHERE job_id = ?", values)
