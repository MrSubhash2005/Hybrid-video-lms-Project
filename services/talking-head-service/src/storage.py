import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import STORAGE_DIR
from src.exceptions import StorageError
from src.logging_config import get_logger

logger = get_logger("storage")

JOB_DB_PATH = STORAGE_DIR / "jobs.db"


def sanitize_filename(filename: str) -> str:
    """Sanitizes filename to prevent path traversal and unsafe characters."""
    base_name = Path(filename).name
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", base_name)

    if not safe_name or safe_name in [".", ".."]:
        safe_name = "upload_file"

    return safe_name


def prepare_job_storage(job_id: str) -> Path:
    """Creates directory structure: storage/jobs/{job_id}/inputs/."""
    job_dir = STORAGE_DIR / job_id / "inputs"

    try:
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir
    except Exception as e:
        logger.error(
            f"Failed to create job storage directory for job {job_id}: {e}"
        )
        raise StorageError(
            f"Failed to initialize storage directory for job '{job_id}'."
        )


def initialize_job_database() -> None:
    """Creates the persistent SQLite job database and schema."""
    try:
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(JOB_DB_PATH) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0,
                    estimated_time_remaining INTEGER,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    output_path TEXT,
                    output_url TEXT,
                    error_message TEXT,
                    voice TEXT,
                    avatar TEXT,
                    gender TEXT
                )
                """
            )
            connection.commit()

    except sqlite3.Error as exc:
        logger.error(f"Failed to initialize job database: {exc}")
        raise StorageError("Failed to initialize job database.")


def create_job(job: Dict[str, Any]) -> None:
    """Creates a persistent job record."""
    initialize_job_database()

    columns = [
        "job_id",
        "status",
        "progress",
        "estimated_time_remaining",
        "created_at",
        "completed_at",
        "output_path",
        "output_url",
        "error_message",
        "voice",
        "avatar",
        "gender",
    ]

    values = [job.get(column) for column in columns]

    try:
        with sqlite3.connect(JOB_DB_PATH) as connection:
            placeholders = ", ".join(["?"] * len(columns))
            column_names = ", ".join(columns)

            connection.execute(
                f"""
                INSERT INTO jobs ({column_names})
                VALUES ({placeholders})
                """,
                values,
            )
            connection.commit()

    except sqlite3.Error as exc:
        logger.error(f"Failed to create job {job.get('job_id')}: {exc}")
        raise StorageError(
            f"Failed to create persistent job '{job.get('job_id')}'."
        )


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Returns a persistent job record or None if it does not exist."""
    initialize_job_database()

    try:
        with sqlite3.connect(JOB_DB_PATH) as connection:
            connection.row_factory = sqlite3.Row

            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()

            if row is None:
                return None

            return dict(row)

    except sqlite3.Error as exc:
        logger.error(f"Failed to read job {job_id}: {exc}")
        raise StorageError(f"Failed to read job '{job_id}'.")


def update_job(job_id: str, **fields: Any) -> None:
    """Updates selected fields of a persistent job record."""
    if not fields:
        return

    initialize_job_database()

    allowed_fields = {
        "status",
        "progress",
        "estimated_time_remaining",
        "created_at",
        "completed_at",
        "output_path",
        "output_url",
        "error_message",
        "voice",
        "avatar",
        "gender",
    }

    updates = {
        key: value
        for key, value in fields.items()
        if key in allowed_fields
    }

    if not updates:
        return

    assignments = ", ".join(f"{key} = ?" for key in updates)
    values = list(updates.values())
    values.append(job_id)

    try:
        with sqlite3.connect(JOB_DB_PATH) as connection:
            cursor = connection.execute(
                f"""
                UPDATE jobs
                SET {assignments}
                WHERE job_id = ?
                """,
                values,
            )
            connection.commit()

            if cursor.rowcount == 0:
                raise StorageError(f"Job '{job_id}' was not found.")

    except sqlite3.Error as exc:
        logger.error(f"Failed to update job {job_id}: {exc}")
        raise StorageError(f"Failed to update job '{job_id}'.")


def save_job_inputs(
    job_id: str,
    image_filename: str,
    image_bytes: bytes,
    audio_filename: str,
    audio_bytes: bytes,
) -> dict:
    """Safely persists validated image and audio files to disk for a job."""
    inputs_dir = prepare_job_storage(job_id)

    safe_img_name = f"image_{sanitize_filename(image_filename)}"
    safe_audio_name = f"audio_{sanitize_filename(audio_filename)}"

    img_path = inputs_dir / safe_img_name
    audio_path = inputs_dir / safe_audio_name

    try:
        with open(img_path, "wb") as f_img:
            f_img.write(image_bytes)

        with open(audio_path, "wb") as f_audio:
            f_audio.write(audio_bytes)

        logger.info(
            f"Saved job inputs for {job_id}: "
            f"image='{img_path}' ({len(image_bytes)} bytes), "
            f"audio='{audio_path}' ({len(audio_bytes)} bytes)"
        )

        return {
            "image_path": str(img_path.resolve()),
            "audio_path": str(audio_path.resolve()),
        }

    except Exception as e:
        logger.error(f"Failed writing input files for job {job_id}: {e}")
        raise StorageError(f"Failed to save input files for job '{job_id}'.")
