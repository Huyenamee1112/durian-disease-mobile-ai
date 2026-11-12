from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "ai" / "data" / "supabase_dataset_4class"
DEFAULT_OUTPUT = REPO_ROOT / "ai" / "data" / "supabase_dataset_4class_processed"

LABELS = (
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


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def pad_to_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = max(width, height)
    # A soft neutral green background keeps padding closer to field images than black/white.
    canvas = Image.new("RGB", (side, side), (188, 205, 184))
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def normalize_image(image: Image.Image, *, output_size: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = pad_to_square(image)
    return image.resize((output_size, output_size), Image.Resampling.LANCZOS)


def augment_image(image: Image.Image, rng: random.Random) -> Image.Image:
    result = image
    if rng.random() < 0.5:
        result = ImageOps.mirror(result)

    angle = rng.uniform(-16, 16)
    result = result.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=(188, 205, 184))

    width, height = result.size
    crop_scale = rng.uniform(0.88, 1.0)
    crop_w = int(width * crop_scale)
    crop_h = int(height * crop_scale)
    left = rng.randint(0, width - crop_w)
    top = rng.randint(0, height - crop_h)
    result = result.crop((left, top, left + crop_w, top + crop_h)).resize(
        (width, height),
        Image.Resampling.LANCZOS,
    )

    result = ImageEnhance.Brightness(result).enhance(rng.uniform(0.86, 1.14))
    result = ImageEnhance.Contrast(result).enhance(rng.uniform(0.88, 1.16))
    result = ImageEnhance.Color(result).enhance(rng.uniform(0.90, 1.12))
    return result


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=92, optimize=True)


def build_processed_dataset(
    *,
    source: Path,
    output: Path,
    output_size: int,
    target_train_per_class: int,
    seed: int,
) -> dict[str, Counter[str]]:
    reset_dir(output)
    rng = random.Random(seed)
    counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}

    normalized_cache: dict[Path, Image.Image] = {}

    for split in SPLITS:
        for label in LABELS:
            files = image_files(source / split / label)
            if not files:
                continue

            normalized_images: list[tuple[Path, Image.Image]] = []
            for source_path in files:
                with Image.open(source_path) as raw_image:
                    image = normalize_image(raw_image, output_size=output_size)
                normalized_cache[source_path] = image
                normalized_images.append((source_path, image))

                target_path = output / split / label / f"{source_path.stem}.jpg"
                save_image(image, target_path)
                counts[split][label] += 1

            if split != "train":
                continue

            needed = max(0, target_train_per_class - len(normalized_images))
            for index in range(needed):
                source_path, base_image = rng.choice(normalized_images)
                augmented = augment_image(base_image, rng)
                target_path = output / split / label / f"{source_path.stem}_aug_{index:03d}.jpg"
                save_image(augmented, target_path)
                counts[split][label] += 1

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize and class-balance the collected 4-class field dataset."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--target-train-per-class", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    counts = build_processed_dataset(
        source=args.source,
        output=args.output,
        output_size=args.output_size,
        target_train_per_class=args.target_train_per_class,
        seed=args.seed,
    )
    for split in SPLITS:
        total = sum(counts[split].values())
        print(f"{split}: total={total}")
        for label in LABELS:
            print(f"  {label}: {counts[split][label]}")
    print(f"Processed dataset: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
