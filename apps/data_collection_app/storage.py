from __future__ import annotations

import os
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from uuid import uuid4

from PIL import Image


APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path("/tmp/durian-data") if os.environ.get("VERCEL") else APP_DIR / "data"
DATA_DIR = Path(os.environ.get("DURIAN_DATA_DIR", DEFAULT_DATA_DIR))
IMAGE_DIR = DATA_DIR / "images"
DATABASE_PATH = DATA_DIR / "submissions.db"
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "durian-submissions")

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


@dataclass(frozen=True)
class SubmissionPayload:
    submission_id: str
    disease: str
    image_path: str
    created_at: str
    tree_stage: str
    notes: str
    latitude: float | None
    longitude: float | None
    location_accuracy: float | None
    location_name: str
    captured_at: str
    status: str = "submitted"


class StorageError(RuntimeError):
    pass


def _supabase_enabled() -> bool:
    return bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY and SUPABASE_BUCKET)


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


def _image_to_jpeg_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=95, optimize=True)
    return buffer.getvalue()


def _request_supabase(
    *,
    path: str,
    method: str,
    body: bytes | None = None,
    content_type: str = "application/json",
    extra_headers: dict[str, str] | None = None,
) -> bytes:
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": content_type,
    }
    if extra_headers:
        headers.update(extra_headers)

    request = Request(
        f"{SUPABASE_URL}{path}",
        data=body,
        method=method,
        headers=headers,
    )
    try:
        with urlopen(request, timeout=20) as response:
            return response.read()
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise StorageError(f"Supabase trả lỗi {error.code}: {detail}") from error
    except URLError as error:
        raise StorageError(f"Không kết nối được Supabase: {error.reason}") from error


def _upload_to_supabase_storage(*, image_bytes: bytes, image_path: str) -> None:
    bucket = quote(SUPABASE_BUCKET, safe="")
    object_path = quote(image_path, safe="/")
    _request_supabase(
        path=f"/storage/v1/object/{bucket}/{object_path}",
        method="POST",
        body=image_bytes,
        content_type="image/jpeg",
        extra_headers={"x-upsert": "false"},
    )


def _insert_supabase_metadata(payload: SubmissionPayload) -> None:
    record: dict[str, Any] = {
        "id": payload.submission_id,
        "disease": payload.disease,
        "image_path": payload.image_path,
        "created_at": payload.created_at,
        "tree_stage": payload.tree_stage,
        "notes": payload.notes,
        "latitude": payload.latitude,
        "longitude": payload.longitude,
        "location_accuracy": payload.location_accuracy,
        "location_name": payload.location_name,
        "captured_at": payload.captured_at or None,
        "status": payload.status,
    }

    try:
        _insert_supabase_record(record)
    except StorageError as error:
        if "location_name" not in str(error):
            raise
        record.pop("location_name", None)
        _insert_supabase_record(record)


def _insert_supabase_record(record: dict[str, Any]) -> None:
    _request_supabase(
        path="/rest/v1/submissions",
        method="POST",
        body=json.dumps(record).encode("utf-8"),
        extra_headers={"Prefer": "return=minimal"},
    )


def _save_to_supabase(*, image: Image.Image, payload: SubmissionPayload) -> None:
    image_bytes = _image_to_jpeg_bytes(image)
    _upload_to_supabase_storage(image_bytes=image_bytes, image_path=payload.image_path)
    try:
        _insert_supabase_metadata(payload)
    except StorageError:
        # Keep the uploaded image instead of deleting it; the path is deterministic
        # so an operator can reconcile orphaned images if metadata insert fails.
        raise


def _build_payload(
    *,
    disease: str,
    tree_stage: str | None,
    notes: str | None,
    latitude: str | None,
    longitude: str | None,
    location_accuracy: str | None,
    location_name: str | None,
    captured_at: str | None,
) -> SubmissionPayload:
    submission_id = str(uuid4())
    return SubmissionPayload(
        submission_id=submission_id,
        disease=disease,
        image_path=f"raw/{submission_id}.jpg",
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        tree_stage=_clean_text(tree_stage, max_length=120),
        notes=_clean_text(notes, max_length=600),
        latitude=_clean_float(latitude),
        longitude=_clean_float(longitude),
        location_accuracy=_clean_float(location_accuracy),
        location_name=_clean_text(location_name, max_length=300),
        captured_at=_clean_text(captured_at, max_length=80),
    )


def _save_to_sqlite(*, image: Image.Image, payload: SubmissionPayload) -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    image_path = IMAGE_DIR / f"{payload.submission_id}.jpg"
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
                payload.submission_id,
                payload.disease,
                str(image_path.relative_to(DATA_DIR)),
                payload.created_at,
                payload.tree_stage,
                payload.notes,
                payload.latitude,
                payload.longitude,
                payload.location_accuracy,
                payload.location_name,
                payload.captured_at,
                payload.status,
            ),
        )


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
    payload = _build_payload(
        disease=disease,
        tree_stage=tree_stage,
        notes=notes,
        latitude=latitude,
        longitude=longitude,
        location_accuracy=location_accuracy,
        location_name=location_name,
        captured_at=captured_at,
    )

    if _supabase_enabled():
        _save_to_supabase(image=image, payload=payload)
    else:
        _save_to_sqlite(image=image, payload=payload)

    return payload.submission_id
