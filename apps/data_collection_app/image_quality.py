from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from PIL import Image, ImageChops, ImageFilter, ImageStat, UnidentifiedImageError


@dataclass(frozen=True)
class ImageQualityResult:
    acceptable: bool
    message: str
    image: Image.Image | None = None


def inspect_image(content: bytes) -> ImageQualityResult:
    """Decode an image and reject samples that are not useful leaf photos."""
    try:
        image = Image.open(BytesIO(content))
        image.load()
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError):
        return ImageQualityResult(False, "Không đọc được ảnh.")

    if min(image.size) < 480:
        return ImageQualityResult(
            False, "Ảnh có độ phân giải thấp, hãy chụp gần lá hơn."
        )

    preview = image.copy()
    preview.thumbnail((192, 192))
    gray = preview.convert("L")
    average_light = ImageStat.Stat(gray).mean[0]
    if average_light < 35:
        return ImageQualityResult(False, "Ảnh quá tối, hãy chụp nơi đủ sáng.")
    if average_light > 240:
        return ImageQualityResult(False, "Ảnh bị chói, hãy đổi góc chụp.")

    detail_preview = image.copy()
    detail_preview.thumbnail((512, 512))
    detail_gray = detail_preview.convert("L")
    high_frequency = ImageChops.difference(
        detail_gray, detail_gray.filter(ImageFilter.GaussianBlur(radius=2))
    )
    high_frequency_stat = ImageStat.Stat(high_frequency)
    sharpness_score = high_frequency_stat.mean[0] + high_frequency_stat.var[0] ** 0.5
    sharpened = detail_gray.filter(
        ImageFilter.UnsharpMask(radius=2, percent=180, threshold=3)
    )
    local_detail = ImageStat.Stat(ImageChops.difference(detail_gray, sharpened)).mean[0]
    tile_scores = _sharp_tile_scores(detail_gray)
    sharp_tile_count = sum(score >= 1.2 for score in tile_scores)
    strong_tile_count = sum(score >= 1.8 for score in tile_scores)
    average_tile_score = sum(tile_scores) / max(len(tile_scores), 1)
    edge_variance = ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES)).var[0]
    if (
        edge_variance < 24
        or sharpness_score < 2.2
        or local_detail < 0.55
        or average_tile_score < 1.0
        or sharp_tile_count < 6
        or strong_tile_count < 5
    ):
        return ImageQualityResult(
            False,
            "Ảnh chưa đủ nét. Hãy chạm vào lá để lấy nét, giữ chắc điện thoại rồi chụp lại.",
        )

    if not _looks_like_leaf_photo(preview):
        return ImageQualityResult(
            False,
            "Ảnh chưa giống ảnh lá sầu riêng. Hãy chụp cận lá, để lá chiếm phần lớn khung hình.",
        )

    return ImageQualityResult(True, "Ảnh đạt yêu cầu.", image)


def _sharp_tile_scores(image: Image.Image) -> list[float]:
    width, height = image.size
    tile_width = max(width // 4, 1)
    tile_height = max(height // 4, 1)
    scores = []

    for top in range(0, height, tile_height):
        for left in range(0, width, tile_width):
            tile = image.crop(
                (left, top, min(left + tile_width, width), min(top + tile_height, height))
            )
            high_frequency = ImageChops.difference(
                tile, tile.filter(ImageFilter.GaussianBlur(radius=1.6))
            )
            stat = ImageStat.Stat(high_frequency)
            scores.append(stat.mean[0] + stat.var[0] ** 0.5)

    return scores


def _looks_like_leaf_photo(image: Image.Image) -> bool:
    rgb_image = image.convert("RGB")
    hsv_image = image.convert("HSV")
    pixels = list(rgb_image.getdata())
    hsv_pixels = list(hsv_image.getdata())
    total = max(len(pixels), 1)

    green_pixels = 0
    leaf_warm_pixels = 0
    skin_pixels = 0
    sky_pixels = 0

    for (red, green, blue), (hue, saturation, value) in zip(pixels, hsv_pixels):
        if 45 <= hue <= 115 and saturation >= 35 and value >= 35:
            green_pixels += 1
        elif 18 <= hue < 45 and saturation >= 55 and value >= 60:
            leaf_warm_pixels += 1

        if (
            red > 95
            and green > 45
            and blue > 25
            and red > green
            and red > blue
            and abs(red - green) > 12
        ):
            skin_pixels += 1

        if blue > 110 and blue > red * 1.15 and blue > green * 1.05:
            sky_pixels += 1

    green_ratio = green_pixels / total
    leaf_ratio = (green_pixels + leaf_warm_pixels) / total
    skin_or_sky_ratio = (skin_pixels + sky_pixels) / total

    if skin_or_sky_ratio > 0.35:
        return False
    return green_ratio >= 0.18 and leaf_ratio >= 0.30
