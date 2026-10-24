from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from urllib.parse import quote
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4


def save_history(
    prediction: dict[str, object],
    *,
    image_bytes: bytes | None = None,
    content_type: str = "image/jpeg",
) -> bool:
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    table = os.environ.get("SUPABASE_HISTORY_TABLE", "diagnosis_history")
    bucket = os.environ.get("SUPABASE_DIAGNOSIS_BUCKET", "durian-diagnosis")
    if not url or not key:
        return False
    diagnosis_id = str(uuid4())
    image_path = f"diagnoses/{diagnosis_id}" if image_bytes else None
    record = {
        "id": diagnosis_id,
        "predicted_label": prediction["label"],
        "disease_name": prediction["disease_name"],
        "confidence": prediction["confidence"],
        "model_name": prediction["model"],
        "image_path": image_path,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    common_headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    if image_bytes and image_path:
        upload_request = Request(
            f"{url}/storage/v1/object/{quote(bucket, safe='')}/{quote(image_path, safe='/')}",
            data=image_bytes,
            method="POST",
            headers={**common_headers, "Content-Type": content_type, "x-upsert": "false"},
        )
        try:
            with urlopen(upload_request, timeout=20):
                pass
        except (HTTPError, URLError):
            return False
    request = Request(
        f"{url}/rest/v1/{table}",
        data=json.dumps(record).encode("utf-8"),
        method="POST",
        headers={
            **common_headers,
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
    )
    try:
        with urlopen(request, timeout=15):
            return True
    except (HTTPError, URLError):
        return False
