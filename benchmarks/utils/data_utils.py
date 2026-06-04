import json
import logging
import os
import pathlib
import re
from collections import defaultdict
from typing import Any

from utils.config_utils import get_config_value, require_config_value

logger = logging.getLogger(__name__)

BENCHMARKS_ROOT = pathlib.Path(__file__).resolve().parents[1]
LONGDOCURL_DEFAULT_SHARD = "4000-4999"
PAGE_NO_PATTERN = re.compile(r"_(\d+)\.[^.]+$")
OCR_TEXT_TEMPLATE = "page_no: {}\n{}\n\n"


def benchmark_data_root(benchmark_name: str) -> pathlib.Path:
    return BENCHMARKS_ROOT / str(benchmark_name) / "data"


def benchmark_cache_root(benchmark_name: str, cache_name: str) -> pathlib.Path:
    return benchmark_data_root(benchmark_name) / "cache" / str(cache_name)


def colpali_pdf_embeddings_path(
    benchmark_name: str,
    stem: Any,
    *,
    shard: str = LONGDOCURL_DEFAULT_SHARD,
) -> pathlib.Path:
    root = benchmark_cache_root(benchmark_name, "colpali") / "pdf_embeddings"
    if benchmark_name == "longdocurl":
        root = root / shard
    return root / f"{stem}.safetensors"


def colpali_question_embeddings_path(benchmark_name: str, question_id: Any) -> pathlib.Path:
    return benchmark_cache_root(benchmark_name, "colpali") / "question_embeddings" / f"{question_id}.safetensors"


def bgem3_cache_root(benchmark_name: str, cache_kind: str | None = None) -> pathlib.Path:
    root = benchmark_cache_root(benchmark_name, "bgem3")
    return root / cache_kind if cache_kind else root


def colbertv2_cache_root(benchmark_name: str, cache_kind: str | None = None) -> pathlib.Path:
    root = benchmark_cache_root(benchmark_name, "colbertv2")
    return root / cache_kind if cache_kind else root


def longdocurl_mineru_dir(shard: str = LONGDOCURL_DEFAULT_SHARD) -> pathlib.Path:
    return benchmark_data_root("longdocurl") / "processed" / "pdfs_mineru" / shard


def load_mmlongbench_samples(input_path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_longdocurl_samples(
    qa_file: str | os.PathLike[str],
    image_prefix: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    samples = []
    with open(qa_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            sample = json.loads(line)
            sample.setdefault("question_id", idx)
            if image_prefix is not None:
                images = []
                for image_path in sample.get("images", []):
                    images.append(str(pathlib.Path(image_prefix) / "/".join(str(image_path).split("/")[-2:])))
                sample["images"] = images
            samples.append(sample)
    return samples


def mmlongbench_file_id(doc_id: Any) -> str:
    filename = pathlib.PurePosixPath(str(doc_id)).name
    file_id = re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", file_id).strip("._") or "document"


def mmlongbench_ocr_dir(benchmark_cfg, doc_id: Any) -> str:
    root = get_config_value(benchmark_cfg, "ocr_json_dir")
    if root is None:
        tmp_dir = require_config_value(benchmark_cfg, "tmp_dir")
        root = os.path.join(tmp_dir, "pdf_jsons")
    return os.path.join(str(root), mmlongbench_file_id(doc_id))


def mmlongbench_png_dir(benchmark_cfg, doc_id: Any) -> str:
    root = get_config_value(benchmark_cfg, "pdf_png_dir")
    if root is None:
        tmp_dir = require_config_value(benchmark_cfg, "tmp_dir")
        root = os.path.join(tmp_dir, "pdf_pngs")
    return os.path.join(str(root), mmlongbench_file_id(doc_id))


def mmlongbench_ocr_page_path(benchmark_cfg, doc_id: Any, page_index: int) -> str:
    return os.path.join(mmlongbench_ocr_dir(benchmark_cfg, doc_id), f"page_{page_index + 1:04d}.json")


def mmlongbench_png_page_path(benchmark_cfg, doc_id: Any, page_index: int) -> str:
    resolution = int(require_config_value(benchmark_cfg, "resolution"))
    return os.path.join(
        mmlongbench_png_dir(benchmark_cfg, doc_id),
        f"page_{page_index + 1:04d}_dpi{resolution}.png",
    )


def write_mmlongbench_ocr_cache(pdf, benchmark_cfg, doc_id: Any) -> int:
    output_dir = mmlongbench_ocr_dir(benchmark_cfg, doc_id)
    os.makedirs(output_dir, exist_ok=True)
    max_pages = int(require_config_value(benchmark_cfg, "max_pages"))
    page_count = min(len(pdf), max_pages)
    for page_index in range(page_count):
        page = pdf[page_index]
        page_text = page.get_text("text").strip() or "[EMPTY PAGE]"
        page_payload = {
            "doc_id": doc_id,
            "file_id": mmlongbench_file_id(doc_id),
            "page_index": page_index,
            "page_number": page_index + 1,
            "text": page_text,
        }
        with open(mmlongbench_ocr_page_path(benchmark_cfg, doc_id, page_index), "w", encoding="utf-8") as f:
            json.dump(page_payload, f, ensure_ascii=False, indent=2)
    return page_count


def write_mmlongbench_png_cache(pdf, benchmark_cfg, doc_id: Any) -> int:
    output_dir = mmlongbench_png_dir(benchmark_cfg, doc_id)
    os.makedirs(output_dir, exist_ok=True)
    max_pages = int(require_config_value(benchmark_cfg, "max_pages"))
    resolution = int(require_config_value(benchmark_cfg, "resolution"))
    page_count = min(len(pdf), max_pages)
    for page_index in range(page_count):
        page_path = mmlongbench_png_page_path(benchmark_cfg, doc_id, page_index)
        if os.path.exists(page_path):
            continue
        pixmap = pdf[page_index].get_pixmap(dpi=resolution)
        pixmap.save(page_path)
    return page_count


def record2text_with_layout(record: dict[str, Any]) -> str:
    text = ""
    img_width = record["docInfo"]["pages"][0]["imageWidth"]
    img_height = record["docInfo"]["pages"][0]["imageHeight"]
    for item in record["layouts"]:
        item_type, sub_type = item["type"], item["subType"]
        item_text = item["text"]
        x1y1 = item["pos"][0]
        x2y2 = item["pos"][2]
        box = tuple(float(f"{value:.2f}") for value in (
            x1y1["x"] / img_width,
            x1y1["y"] / img_height,
            x2y2["x"] / img_width,
            x2y2["y"] / img_height,
        ))
        text += f"(type: {item_type}, sub_type: {sub_type}, box: {box}) {item_text}\n"
    return text


def record2text(record: dict[str, Any]) -> str:
    return "".join(f"{item['text']}\n" for item in record["layouts"])


def extract_page_nos_from_images(images) -> list[int]:
    page_nos = []
    seen = set()
    for image_path in images or []:
        match = PAGE_NO_PATTERN.search(os.path.basename(str(image_path)))
        if match is None:
            continue
        page_no = int(match.group(1))
        if page_no not in seen:
            seen.add(page_no)
            page_nos.append(page_no)
    return page_nos


def load_pymupdf_record(doc_no: str, ocr_json_dir: str | os.PathLike[str]) -> dict[str, Any]:
    root = pathlib.Path(ocr_json_dir)
    direct_path = root / f"{doc_no}.json"
    nested_path = root / doc_no[:4] / f"{doc_no}.json"
    json_path = direct_path if direct_path.exists() else nested_path
    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_page_texts_from_contents(contents: list[dict[str, Any]], selected_pages: list[int]) -> list[tuple[int, str]]:
    selected_page_set = set(selected_pages)
    page_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for item in contents:
        page_no = item["page_no"]
        if page_no not in selected_page_set:
            continue
        page_map[page_no][item["block_no"]][item["line_no"]].append(item)

    page_texts = []
    for page_no in selected_pages:
        if page_no not in page_map:
            continue
        lines = []
        for block_no in sorted(page_map[page_no]):
            for line_no in sorted(page_map[page_no][block_no]):
                words = sorted(page_map[page_no][block_no][line_no], key=lambda word: word["word_no"])
                line_text = " ".join(word["word"] for word in words if word["word"])
                if line_text:
                    lines.append(line_text)
        page_text = "\n".join(lines).strip()
        if page_text:
            page_texts.append((page_no, page_text))
    return page_texts


def get_pure_ocr_prompt_pymupdf(doc_no: str, images=None, ocr_json_dir=None, **kwargs):
    if ocr_json_dir is None:
        raise ValueError("ocr_json_dir is required for PyMuPDF OCR evaluation")

    selected_pages = extract_page_nos_from_images(images or [])
    if not selected_pages:
        start_page = kwargs.pop("start_page", 0)
        end_page = kwargs.pop("end_page", start_page + 1)
        selected_pages = list(range(start_page, end_page + 1))

    record = load_pymupdf_record(doc_no, ocr_json_dir)
    page_texts = build_page_texts_from_contents(record["contents"], selected_pages)
    pages_used = [page_no for page_no, _ in page_texts]
    logger.debug("number of pages used: %s", len(pages_used))

    ocr_prompt = "\n\n"
    for page_no, page_text in page_texts:
        ocr_prompt += OCR_TEXT_TEMPLATE.format(page_no + 1, page_text)
    return ocr_prompt, pages_used
