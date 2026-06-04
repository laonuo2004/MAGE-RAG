import json
from pathlib import Path
from typing import Any


def load_mineru_document(mineru_dir: Path) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]], Path]:
    layout_path = mineru_dir / "layout.json"
    if not layout_path.exists():
        raise FileNotFoundError(f"Missing MinerU layout.json: {layout_path}")
    with layout_path.open("r", encoding="utf-8") as f:
        layout = json.load(f)
    pages = layout.get("pdf_info")
    if not isinstance(pages, list):
        raise ValueError(f"MinerU layout.json missing list pdf_info: {layout_path}")

    content_path = find_content_list_v2(mineru_dir)
    with content_path.open("r", encoding="utf-8") as f:
        content_pages = json.load(f)
    if not isinstance(content_pages, list):
        raise ValueError(f"MinerU content_list_v2 must be a page list: {content_path}")
    return pages, content_pages, content_path


def find_content_list_v2(mineru_dir: Path) -> Path:
    candidates = sorted(mineru_dir.glob("*_content_list_v2.json"))
    direct = mineru_dir / "content_list_v2.json"
    if direct.exists():
        return direct
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Missing *_content_list_v2.json under {mineru_dir}")

