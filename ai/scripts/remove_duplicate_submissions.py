from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from download_supabase_dataset import get_required_env, load_env_file


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPO_ROOT / "ai" / "data" / "supabase_dataset_20260922"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class SupabaseDeleteError(RuntimeError):
    pass


def request(
    *,
    base_url: str,
    service_role_key: str,
    path: str,
    method: str,
    headers: dict[str, str] | None = None,
) -> bytes:
    request_headers = {
        "apikey": service_role_key,
        "Authorization": f"Bearer {service_role_key}",
    }
    if headers:
        request_headers.update(headers)
    try:
        with urlopen(Request(base_url + path, headers=request_headers, method=method), timeout=30) as response:
            return response.read()
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise SupabaseDeleteError(f"Supabase returned HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise SupabaseDeleteError(f"Could not connect to Supabase: {error.reason}") from error


def duplicate_targets(dataset_dir: Path) -> list[dict[str, str]]:
    manifest_path = dataset_dir / "manifest.csv"
    rows = {row["id"]: row for row in csv.DictReader(manifest_path.open(encoding="utf-8"))}
    hashes: dict[str, list[Path]] = defaultdict(list)
    for path in dataset_dir.glob("*/*/*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            hashes[hashlib.sha256(path.read_bytes()).hexdigest()].append(path)

    targets: list[dict[str, str]] = []
    for digest, paths in hashes.items():
        if len(paths) < 2:
            continue
        group = sorted((rows[path.stem] for path in paths), key=lambda row: (row["created_at"], row["id"]))
        kept = group[0]
        for duplicate in group[1:]:
            if duplicate["label"] != kept["label"]:
                raise SupabaseDeleteError(
                    f"Identical image has conflicting labels: {kept['id']} and {duplicate['id']}"
                )
            targets.append(
                {
                    "id": duplicate["id"],
                    "image_path": duplicate["image_path"],
                    "label": duplicate["label"],
                    "created_at": duplicate["created_at"],
                    "duplicate_of_id": kept["id"],
                    "sha256": digest,
                }
            )
    return sorted(targets, key=lambda row: (row["created_at"], row["id"]))


def delete_submission(*, base_url: str, service_role_key: str, submission_id: str) -> None:
    query = urlencode({"id": f"eq.{submission_id}"})
    response = request(
        base_url=base_url,
        service_role_key=service_role_key,
        path=f"/rest/v1/submissions?{query}",
        method="DELETE",
        headers={"Prefer": "return=representation", "Accept": "application/json"},
    )
    deleted_rows = json.loads(response.decode("utf-8"))
    if len(deleted_rows) != 1 or deleted_rows[0].get("id") != submission_id:
        raise SupabaseDeleteError(f"Expected to delete one submission {submission_id}, got: {deleted_rows}")


def delete_storage_object(*, base_url: str, service_role_key: str, bucket: str, image_path: str) -> None:
    request(
        base_url=base_url,
        service_role_key=service_role_key,
        path=f"/storage/v1/object/{quote(bucket, safe='')}/{quote(image_path, safe='/')}",
        method="DELETE",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove exact duplicate submissions and their Storage objects.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--execute", action="store_true", help="Perform deletion; default is dry run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = duplicate_targets(args.dataset_dir)
    audit_path = args.dataset_dir / "duplicate_removal_audit.json"
    audit_path.write_text(json.dumps({"targets": targets, "executed": args.execute}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(targets, ensure_ascii=False, indent=2))
    print(f"Found {len(targets)} duplicate submissions. Audit: {audit_path}")
    if not args.execute:
        return 0

    load_env_file(args.env_file)
    base_url, service_role_key, bucket = get_required_env()
    for target in targets:
        delete_submission(base_url=base_url, service_role_key=service_role_key, submission_id=target["id"])
        delete_storage_object(
            base_url=base_url,
            service_role_key=service_role_key,
            bucket=bucket,
            image_path=target["image_path"],
        )
        print(f"Deleted submission and image: {target['id']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SupabaseDeleteError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
