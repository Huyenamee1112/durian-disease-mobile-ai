from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from PIL import Image


APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path("/tmp/durian-data") if os.environ.get("VERCEL") else APP_DIR / "data"
DATA_DIR = Path(os.environ.get("DURIAN_DATA_DIR", DEFAULT_DATA_DIR))
IMAGE_DIR = DATA_DIR / "images"
DATABASE_PATH = DATA_DIR / "submissions.db"

SUBMISSION_COLUMNS = {
    "tree_stage": "TEXT",
    "notes": "TEXT",
    "latitude": "REAL",
    "longitude": "REAL",
    "location_accuracy": "REAL",
    "location_name": "TEXT",
    "captured_at": "TEXT",
    "status": "TEXT NOT NULL DEFAULT 'submitted'",
    "expert_label": "TEXT",
}


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS submissions (
            id TEXT PRIMARY KEY,
            disease TEXT NOT NULL,
            image_path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            tree_stage TEXT,
            notes TEXT,
            latitude REAL,
            longitude REAL,
            location_accuracy REAL,
            location_name TEXT,
            captured_at TEXT,
            status TEXT NOT NULL DEFAULT 'submitted',
            expert_label TEXT
        )
        """
    )
    _ensure_columns(connection)
    return connection


def _ensure_columns(connection: sqlite3.Connection) -> None:
    existing_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(submissions)")
    }
    for column_name, column_type in SUBMISSION_COLUMNS.items():
        if column_name not in existing_columns:
            connection.execute(
                f"ALTER TABLE submissions ADD COLUMN {column_name} {column_type}"
            )


def _clean_text(value: str | None, *, max_length: int) -> str:
    return (value or "").strip()[:max_length]


def _clean_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def save_submission(
    *,
    image: Image.Image,
    disease: str,
    tree_stage: str | None = None,
    notes: str | None = None,
    latitude: str | None = None,
    longitude: str | None = None,
    location_accuracy: str | None = None,
    location_name: str | None = None,
    captured_at: str | None = None,
) -> str:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    submission_id = str(uuid4())
    image_path = IMAGE_DIR / f"{submission_id}.jpg"
    image.convert("RGB").save(image_path, "JPEG", quality=95, optimize=True)

    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO submissions (
                id,
                disease,
                image_path,
                created_at,
                tree_stage,
                notes,
                latitude,
                longitude,
                location_accuracy,
                location_name,
                captured_at,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                submission_id,
                disease,
                str(image_path.relative_to(DATA_DIR)),
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                _clean_text(tree_stage, max_length=120),
                _clean_text(notes, max_length=600),
                _clean_float(latitude),
                _clean_float(longitude),
                _clean_float(location_accuracy),
                _clean_text(location_name, max_length=300),
                _clean_text(captured_at, max_length=80),
                "submitted",
            ),
        )
    return submission_id
