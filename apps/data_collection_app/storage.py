from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from PIL import Image


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DURIAN_DATA_DIR", APP_DIR / "data"))
IMAGE_DIR = DATA_DIR / "images"
DATABASE_PATH = DATA_DIR / "submissions.db"


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
            created_at TEXT NOT NULL
        )
        """
    )
    return connection


def save_submission(*, image: Image.Image, disease: str) -> str:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    submission_id = str(uuid4())
    image_path = IMAGE_DIR / f"{submission_id}.jpg"
    image.convert("RGB").save(image_path, "JPEG", quality=92, optimize=True)

    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO submissions (id, disease, image_path, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                submission_id,
                disease,
                str(image_path.relative_to(DATA_DIR)),
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
    return submission_id
