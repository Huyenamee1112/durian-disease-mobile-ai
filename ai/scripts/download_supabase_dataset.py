from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = REPO_ROOT / ".env"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "ai" / "data" / "supabase_dataset"
LABEL_COLUMNS = ("final_label", "expert_label", "disease")
VALID_SPLITS = {"train", "val", "test"}


class SupabaseDatasetError(RuntimeError):
    pass


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_required_env() -> tuple[str, str, str]:
    supabase_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    service_role_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    bucket = os.environ.get("SUPABASE_BUCKET", "durian-submissions")

    missing = [
        name
        for name, value in (
            ("SUPABASE_URL", supabase_url),
            ("SUPABASE_SERVICE_ROLE_KEY", service_role_key),
            ("SUPABASE_BUCKET", bucket),
        )
        if not value
    ]
    if missing:
        raise SupabaseDatasetError(
            "Missing environment variables: " + ", ".join(missing)
        )
    return supabase_url, service_role_key, bucket


def supabase_request(
    *,
    supabase_url: str,
    service_role_key: str,
    path: str,
    headers: dict[str, str] | None = None,
) -> bytes:
    request_headers = {
        "apikey": service_role_key,
        "Authorization": f"Bearer {service_role_key}",
    }
    if headers:
        request_headers.update(headers)

    request = Request(f"{supabase_url}{path}", headers=request_headers, method="GET")
    try:
        with urlopen(request, timeout=30) as response:
            return response.read()
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise SupabaseDatasetError(
            f"Supabase returned HTTP {error.code}: {detail}"
        ) from error
    except URLError as error:
        raise SupabaseDatasetError(f"Could not connect to Supabase: {error.reason}") from error


def fetch_submissions(
    *,
    supabase_url: str,
    service_role_key: str,
    page_size: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0

    while True:
        request_size = page_size
        if limit is not None:
            remaining = limit - len(rows)
            if remaining <= 0:
                break
            request_size = min(request_size, remaining)

        query = urlencode(
            {
                "select": "*",
                "order": "created_at.desc",
                "limit": str(request_size),
                "offset": str(offset),
            }
        )
        body = supabase_request(
            supabase_url=supabase_url,
            service_role_key=service_role_key,
            path=f"/rest/v1/submissions?{query}",
            headers={"Accept": "application/json"},
        )
        page_rows = json.loads(body.decode("utf-8"))
        if not page_rows:
            break

        rows.extend(page_rows)
        if len(page_rows) < request_size:
            break
        offset += request_size

    return rows


def choose_label(row: dict[str, Any]) -> str:
    for column in LABEL_COLUMNS:
        value = (row.get(column) or "").strip()
        if value:
            return value
    return ""


def filter_rows(
    rows: list[dict[str, Any]],
    *,
    reviewed_only: bool,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    filtered: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()

    for row in rows:
        review_status = (row.get("review_status") or "").strip().lower()
        if review_status == "invalid":
            skipped["invalid"] += 1
            continue
        if reviewed_only and review_status != "valid":
            skipped["not_reviewed"] += 1
            continue
        if not row.get("image_path"):
            skipped["missing_image_path"] += 1
            continue
        label = choose_label(row)
        if not label:
            skipped["missing_label"] += 1
            continue

        row["_label"] = label
        filtered.append(row)

    return filtered, skipped


def auto_split_rows(
    rows: list[dict[str, Any]],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        existing_split = (row.get("split") or "").strip().lower()
        if existing_split in VALID_SPLITS:
            row["_split"] = existing_split
        else:
            grouped[row["_label"]].append(row)

    random_generator = random.Random(seed)
    for label_rows in grouped.values():
        random_generator.shuffle(label_rows)
        total = len(label_rows)
        train_count = round(total * train_ratio)
        val_count = round(total * val_ratio)
        if total >= 3:
            train_count = max(1, min(train_count, total - 2))
            val_count = max(1, min(val_count, total - train_count - 1))
        else:
            train_count = total
            val_count = 0

        for index, row in enumerate(label_rows):
            if index < train_count:
                row["_split"] = "train"
            elif index < train_count + val_count:
                row["_split"] = "val"
            else:
                row["_split"] = "test"


def safe_filename(row: dict[str, Any]) -> str:
    submission_id = (row.get("id") or Path(row["image_path"]).stem).strip()
    suffix = Path(row["image_path"]).suffix or ".jpg"
    return f"{submission_id}{suffix}"


def download_image(
    *,
    supabase_url: str,
    service_role_key: str,
    bucket: str,
    image_path: str,
) -> bytes:
    bucket_path = quote(bucket, safe="")
    object_path = quote(image_path, safe="/")
    return supabase_request(
        supabase_url=supabase_url,
        service_role_key=service_role_key,
        path=f"/storage/v1/object/{bucket_path}/{object_path}",
    )


def write_manifest(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "split",
                "label",
                "disease",
                "image_path",
                "local_path",
                "download_status",
                "download_error",
                "review_status",
                "reject_reason",
                "created_at",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row.get("id", ""),
                    "split": row.get("_split", ""),
                    "label": row.get("_label", ""),
                    "disease": row.get("disease", ""),
                    "image_path": row.get("image_path", ""),
                    "local_path": row.get("_local_path", ""),
                    "download_status": row.get("_download_status", ""),
                    "download_error": row.get("_download_error", ""),
                    "review_status": row.get("review_status", ""),
                    "reject_reason": row.get("reject_reason", ""),
                    "created_at": row.get("created_at", ""),
                }
            )


def print_summary(rows: list[dict[str, Any]], skipped: Counter[str]) -> None:
    split_label_counts: Counter[tuple[str, str]] = Counter()
    label_counts: Counter[str] = Counter()
    for row in rows:
        split = row.get("_split", "")
        label = row.get("_label", "")
        split_label_counts[(split, label)] += 1
        label_counts[label] += 1

    print(f"Usable rows: {len(rows)}")
    if skipped:
        print("Skipped rows:")
        for reason, count in sorted(skipped.items()):
            print(f"  {reason}: {count}")

    print("Label counts:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    print("Split counts:")
    for split in ("train", "val", "test"):
        print(f"  {split}:")
        for label in sorted(label_counts):
            count = split_label_counts[(split, label)]
            if count:
                print(f"    {label}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Supabase submissions into train/val/test class folders."
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--page-size", type=int, default=500)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--reviewed-only",
        action="store_true",
        help="Only export rows with review_status='valid'.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload images even if they already exist locally.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove existing train/val/test folders and manifest before exporting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch metadata and print the planned split without downloading images.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)
    supabase_url, service_role_key, bucket = get_required_env()

    rows = fetch_submissions(
        supabase_url=supabase_url,
        service_role_key=service_role_key,
        page_size=args.page_size,
        limit=args.limit,
    )
    usable_rows, skipped = filter_rows(rows, reviewed_only=args.reviewed_only)
    auto_split_rows(
        usable_rows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    if args.dry_run:
        print_summary(usable_rows, skipped)
        print(f"Dry run only. Output dir would be: {args.output_dir}")
        return 0

    if args.clean_output and args.output_dir.exists():
        for child_name in ("train", "val", "test", "manifest.csv"):
            child_path = args.output_dir / child_name
            if child_path.is_dir():
                shutil.rmtree(child_path)
            elif child_path.exists():
                child_path.unlink()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(usable_rows, start=1):
        split = row["_split"]
        label = row["_label"]
        destination = args.output_dir / split / label / safe_filename(row)
        destination.parent.mkdir(parents=True, exist_ok=True)
        row["_local_path"] = str(destination.relative_to(args.output_dir))

        if destination.exists() and not args.force:
            row["_download_status"] = "existing"
            row["_download_error"] = ""
            continue

        try:
            image_bytes = download_image(
                supabase_url=supabase_url,
                service_role_key=service_role_key,
                bucket=bucket,
                image_path=row["image_path"],
            )
        except SupabaseDatasetError as error:
            row["_download_status"] = "error"
            row["_download_error"] = str(error)
            skipped["download_error"] += 1
            print(
                f"Skipping missing/unreadable object {row['image_path']}: {error}",
                flush=True,
            )
            continue

        destination.write_bytes(image_bytes)
        row["_download_status"] = "downloaded"
        row["_download_error"] = ""
        if index % 25 == 0:
            print(f"Downloaded {index}/{len(usable_rows)} images...", flush=True)

    write_manifest(args.output_dir, usable_rows)
    print_summary(usable_rows, skipped)
    print(f"Dataset exported to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SupabaseDatasetError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
