from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from PIL import Image, ImageFilter, ImageStat, UnidentifiedImageError


@dataclass(frozen=True)
class ImageQualityResult:
    acceptable: bool
    message: str
    image: Image.Image | None = None


def inspect_image(content: bytes) -> ImageQualityResult:
    """Decode an image and reject only clearly unusable samples."""
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

    edge_variance = ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES)).var[0]
    if edge_variance < 20:
        return ImageQualityResult(False, "Ảnh có thể bị mờ, hãy giữ chắc điện thoại.")

    return ImageQualityResult(True, "Ảnh đạt yêu cầu.", image)
