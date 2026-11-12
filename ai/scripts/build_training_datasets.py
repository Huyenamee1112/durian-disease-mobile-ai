from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIELD_SOURCE = REPO_ROOT / "ai" / "data" / "supabase_dataset"
DEFAULT_PUBLIC_SOURCE = REPO_ROOT / "ai" / "data" / "Durian_Leaf_Diseases"
DEFAULT_FIELD_4CLASS = REPO_ROOT / "ai" / "data" / "supabase_dataset_4class"
DEFAULT_COMBINED_4CLASS = REPO_ROOT / "ai" / "data" / "combined_public_field_4class"

FOUR_CLASS_LABELS = (
    "Leaf_Algal",
    "Leaf_Blight",
    "Leaf_Colletotrichum",
    "Leaf_Healthy",
)
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def reset_dir(directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def copy_images(source_dir: Path, target_dir: Path, *, prefix: str = "") -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for source in image_files(source_dir):
        target_name = f"{prefix}{source.name}" if prefix else source.name
        shutil.copy2(source, target_dir / target_name)
        copied += 1
    return copied


def build_field_4class(source: Path, output: Path) -> dict[str, dict[str, int]]:
    reset_dir(output)
    counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        counts[split] = {}
        for label in FOUR_CLASS_LABELS:
            copied = copy_images(source / split / label, output / split / label)
            counts[split][label] = copied
    return counts


def build_balanced_field_4class(
    source: Path,
    output: Path,
    *,
    per_class: int,
    seed: int,
) -> tuple[dict[str, dict[str, int]], list[dict[str, str]]]:
    """Sample an equal number of unique field images and stratify new splits."""
    grouped: dict[str, list[Path]] = {label: [] for label in FOUR_CLASS_LABELS}
    seen: dict[str, tuple[str, Path]] = {}
    removed: list[dict[str, str]] = []

    for split in SPLITS:
        for label in FOUR_CLASS_LABELS:
            for path in image_files(source / split / label):
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in seen:
                    previous_label, previous_path = seen[digest]
                    if previous_label != label:
                        raise ValueError(f"Conflicting labels for identical images: {previous_path}, {path}")
                    removed.append(
                        {
                            "removed": str(path.relative_to(source)),
                            "kept": str(previous_path.relative_to(source)),
                            "sha256": digest,
                        }
                    )
                    continue
                seen[digest] = (label, path)
                grouped[label].append(path)

    reset_dir(output)
    rng = random.Random(seed)
    counts: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    for label, paths in grouped.items():
        if len(paths) < per_class:
            raise ValueError(f"{label} has only {len(paths)} unique images; need {per_class}.")
        rng.shuffle(paths)
        selected = paths[:per_class]
        train_count = round(per_class * 0.70)
        val_count = round(per_class * 0.15)
        assignments = {
            "train": selected[:train_count],
            "val": selected[train_count : train_count + val_count],
            "test": selected[train_count + val_count :],
        }
        for split, paths_in_split in assignments.items():
            counts[split][label] = 0
            for path in paths_in_split:
                destination = output / split / label / path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                counts[split][label] += 1
    return counts, removed


def build_combined_4class(
    *,
    public_source: Path,
    field_source: Path,
    output: Path,
    field_eval_only: bool = False,
) -> dict[str, dict[str, int]]:
    reset_dir(output)
    counts: dict[str, dict[str, int]] = {}

    for split in SPLITS:
        counts[split] = {}
        for label in FOUR_CLASS_LABELS:
            target_dir = output / split / label
            field_count = copy_images(field_source / split / label, target_dir, prefix="field_")
            public_count = 0
            if split == "train" or (split == "val" and not field_eval_only):
                public_count = copy_images(
                    public_source / split / label,
                    target_dir,
                    prefix="public_",
                )
            counts[split][label] = field_count + public_count
    return counts


def summarize_counts(counts: dict[str, dict[str, int]]) -> dict[str, int]:
    total_by_label: Counter[str] = Counter()
    for split_counts in counts.values():
        total_by_label.update(split_counts)
    return dict(total_by_label)


def deduplicate_dataset(
    output: Path,
    *,
    exclude_conflicting_labels: bool = False,
) -> list[dict[str, str]]:
    """Keep evaluation copies first so no identical image leaks into training."""
    groups: dict[str, list[Path]] = {}
    for split in SPLITS:
        for label in FOUR_CLASS_LABELS:
            for path in image_files(output / split / label):
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                groups.setdefault(digest, []).append(path)

    removed: list[dict[str, str]] = []
    priority = {"test": 0, "val": 1, "train": 2}
    for digest, paths in groups.items():
        if len(paths) < 2:
            continue
        labels = {path.parent.name for path in paths}
        if len(labels) > 1:
            if not exclude_conflicting_labels:
                raise ValueError(f"Conflicting labels for identical images: {', '.join(map(str, paths))}")
            for path in paths:
                removed.append(
                    {
                        "removed": str(path.relative_to(output)),
                        "kept": "",
                        "sha256": digest,
                        "reason": "conflicting_labels",
                    }
                )
            continue

        ordered_paths = sorted(paths, key=lambda path: (priority[path.parts[-3]], str(path)))
        kept = ordered_paths[0]
        for path in ordered_paths[1:]:
            removed.append(
                {
                    "removed": str(path.relative_to(output)),
                    "kept": str(kept.relative_to(output)),
                    "sha256": digest,
                }
            )

    for entry in removed:
        (output / entry["removed"]).unlink()
    return removed


def deduplicate_field(output: Path) -> list[dict[str, str]]:
    return deduplicate_dataset(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build clean 4-class field and public+field training datasets."
    )
    parser.add_argument("--field-source", type=Path, default=DEFAULT_FIELD_SOURCE)
    parser.add_argument("--public-source", type=Path, default=DEFAULT_PUBLIC_SOURCE)
    parser.add_argument("--field-output", type=Path, default=DEFAULT_FIELD_4CLASS)
    parser.add_argument("--combined-output", type=Path, default=DEFAULT_COMBINED_4CLASS)
    parser.add_argument("--field-only", action="store_true")
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument(
        "--deduplicate-combined",
        action="store_true",
        help="Remove exact duplicate images across train, validation, and test in the combined dataset.",
    )
    parser.add_argument(
        "--exclude-conflicting-combined-labels",
        action="store_true",
        help="Remove every copy of an exact duplicate that has conflicting labels in the combined dataset.",
    )
    parser.add_argument(
        "--combined-field-eval-only",
        action="store_true",
        help="Use field images only for combined validation and test; public images are used only for train.",
    )
    parser.add_argument(
        "--balanced-per-class",
        type=int,
        help="Create a new stratified field dataset with this many unique images per class.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.balanced_per_class:
        field_counts, removed = build_balanced_field_4class(
            args.field_source,
            args.field_output,
            per_class=args.balanced_per_class,
            seed=args.seed,
        )
    else:
        field_counts = build_field_4class(args.field_source, args.field_output)
        removed = deduplicate_field(args.field_output) if args.deduplicate else []
    if args.deduplicate and not args.balanced_per_class:
        field_counts = {split: {label: len(image_files(args.field_output / split / label)) for label in FOUR_CLASS_LABELS} for split in SPLITS}

    report = {
        "labels": list(FOUR_CLASS_LABELS),
        "field_4class": {
            "path": str(args.field_output),
            "splits": field_counts,
            "total_by_label": summarize_counts(field_counts),
        },
        "duplicate_images_removed": removed,
    }
    if not args.field_only:
        combined_counts = build_combined_4class(
            public_source=args.public_source,
            field_source=args.field_output,
            output=args.combined_output,
            field_eval_only=args.combined_field_eval_only,
        )
        combined_removed = []
        if args.deduplicate_combined:
            combined_removed = deduplicate_dataset(
                args.combined_output,
                exclude_conflicting_labels=args.exclude_conflicting_combined_labels,
            )
            combined_counts = {
                split: {
                    label: len(image_files(args.combined_output / split / label))
                    for label in FOUR_CLASS_LABELS
                }
                for split in SPLITS
            }
        report["combined_public_field_4class"] = {
            "path": str(args.combined_output),
            "splits": combined_counts,
            "total_by_label": summarize_counts(combined_counts),
            "note": (
                "Combined train uses public+field; validation/test use field images only."
                if args.combined_field_eval_only
                else "Combined train/val use public+field images; test uses field images only."
            ),
            "duplicate_images_removed": combined_removed,
        }
    report_path = (args.field_output if args.field_only else args.combined_output) / "dataset_build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
