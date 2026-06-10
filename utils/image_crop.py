from __future__ import annotations

from typing import Any


def normalized_bbox_1000_to_pixel_box(
    bbox: Any,
    image_size: tuple[int, int],
    *,
    padding_ratio: float = 0.06,
    min_padding: int = 8,
    min_size: int = 20,
) -> tuple[int, int, int, int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    width, height = image_size
    try:
        x0, y0, x1, y1 = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    left = int(round(x0 / 1000.0 * width))
    top = int(round(y0 / 1000.0 * height))
    right = int(round(x1 / 1000.0 * width))
    bottom = int(round(y1 / 1000.0 * height))
    if right <= left or bottom <= top:
        return None
    pad_x = max(min_padding, int((right - left) * padding_ratio))
    pad_y = max(min_padding, int((bottom - top) * padding_ratio))
    left = max(0, left - pad_x)
    top = max(0, top - pad_y)
    right = min(width, right + pad_x)
    bottom = min(height, bottom + pad_y)
    if right - left < min_size or bottom - top < min_size:
        return None
    return left, top, right, bottom


def crop_image_to_normalized_bbox_1000(image: Any, bbox: Any):
    box = normalized_bbox_1000_to_pixel_box(bbox, image.size)
    if box is None:
        return None
    return image.crop(box)
