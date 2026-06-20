#!/usr/bin/env python3
import argparse
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DPI = 144
logger = logging.getLogger("preprocess_documents")


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    input_path: Path
    pdf_dir: Path
    png_dir: Path
    ocr_dir: Path
    default_max_pages: int | None


BENCHMARK_SPECS = {
    "longdocurl": BenchmarkSpec(
        name="longdocurl",
        input_path=CODE_DIR / "benchmarks/longdocurl/data/raw/LongDocURL.jsonl",
        pdf_dir=CODE_DIR / "benchmarks/longdocurl/data/raw/pdfs/4000-4999",
        png_dir=CODE_DIR / "benchmarks/longdocurl/data/processed/pdf_pngs/4000-4999",
        ocr_dir=CODE_DIR / "benchmarks/longdocurl/data/processed/pdf_jsons/4000-4999",
        default_max_pages=None,
    ),
    "mmlongbench": BenchmarkSpec(
        name="mmlongbench",
        input_path=CODE_DIR / "benchmarks/mmlongbench/data/raw/samples.json",
        pdf_dir=CODE_DIR / "benchmarks/mmlongbench/data/raw/documents",
        png_dir=CODE_DIR / "benchmarks/mmlongbench/data/processed/pdf_pngs",
        ocr_dir=CODE_DIR / "benchmarks/mmlongbench/data/processed/pdf_jsons",
        default_max_pages=120,
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Render benchmark PDFs into page PNG files.")
    parser.add_argument("--benchmark", choices=sorted(BENCHMARK_SPECS), required=True)
    parser.add_argument("--input-path", default=None, help="Override the benchmark QA input path.")
    parser.add_argument("--pdf-dir", default=None, help="Override the benchmark PDF directory.")
    parser.add_argument("--pdf-png-dir", default=None, help="Override the output PNG root.")
    parser.add_argument("--ocr-json-dir", default=None, help="Override the output OCR JSON root.")
    parser.add_argument("--mode", choices=["image", "ocr", "both"], default="image")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="PNG render DPI.")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum pages per PDF. Defaults to all pages for LongDocURL and 120 for MMLongBench.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of PDFs to render concurrently.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N selected documents.")
    parser.add_argument("--doc-id", action="append", default=None, help="Specific document id to process. Can be repeated.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate all selected PNG pages.")
    return parser.parse_args()


def resolve_paths(args):
    spec = BENCHMARK_SPECS[args.benchmark]
    return (
        Path(args.input_path or spec.input_path),
        Path(args.pdf_dir or spec.pdf_dir),
        Path(args.pdf_png_dir or spec.png_dir),
        Path(args.ocr_json_dir or spec.ocr_dir),
        spec.default_max_pages if args.max_pages is None else args.max_pages,
    )


def load_doc_ids(benchmark, input_path, explicit_doc_ids=None):
    if benchmark == "longdocurl":
        with Path(input_path).open("r", encoding="utf-8") as file:
            available = {str(json.loads(line)["doc_no"]) for line in file if line.strip()}
    else:
        with Path(input_path).open("r", encoding="utf-8") as file:
            available = {str(sample["doc_id"]) for sample in json.load(file)}

    if not explicit_doc_ids:
        return sorted(available)

    selected = sorted(set(str(doc_id) for doc_id in explicit_doc_ids))
    missing = sorted(set(selected) - available)
    if missing:
        raise ValueError(f"Unknown {benchmark} document ids: {missing}")
    return selected


def document_key(benchmark, doc_id):
    if benchmark == "longdocurl":
        return str(doc_id)
    filename = Path(str(doc_id)).name
    stem = re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "document"


def pdf_path(benchmark, pdf_dir, doc_id):
    filename = f"{doc_id}.pdf" if benchmark == "longdocurl" else str(doc_id)
    return Path(pdf_dir) / filename


def png_page_path(benchmark, png_root, doc_id, page_index, dpi):
    doc_key = document_key(benchmark, doc_id)
    if benchmark == "longdocurl":
        return Path(png_root) / doc_key[:4] / f"{doc_key}_{page_index}.png"
    return Path(png_root) / doc_key / f"page_{page_index + 1:04d}_dpi{dpi}.png"


def expected_page_size(page, dpi):
    import fitz

    scale = dpi / 72
    pixel_rect = (page.rect * fitz.Matrix(scale, scale)).irect
    return pixel_rect.width, pixel_rect.height


def page_needs_render(output_path, page, dpi, overwrite):
    if overwrite or not output_path.exists():
        return True
    import fitz

    existing = fitz.Pixmap(output_path)
    return (existing.width, existing.height) != expected_page_size(page, dpi)


def write_mmlongbench_ocr_page(ocr_root, doc_id, page_index, page, overwrite):
    doc_key = document_key("mmlongbench", doc_id)
    output_path = Path(ocr_root) / doc_key / f"page_{page_index + 1:04d}.json"
    if output_path.exists() and not overwrite:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "doc_id": str(doc_id),
        "file_id": doc_key,
        "page_index": page_index,
        "page_number": page_index + 1,
        "text": page.get_text("text").strip() or "[EMPTY PAGE]",
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def write_longdocurl_ocr(ocr_root, doc_id, source_path, pdf, page_count, dpi, overwrite):
    doc_no = str(doc_id)
    output_path = Path(ocr_root) / doc_no[:4] / f"{doc_no}.json"
    if output_path.exists() and not overwrite:
        return False

    contents = []
    image_size = None
    for page_index in range(page_count):
        page = pdf[page_index]
        if image_size is None:
            image_size = list(expected_page_size(page, dpi))
        width = max(float(page.rect.width), 1)
        height = max(float(page.rect.height), 1)
        for x0, y0, x1, y1, word, block_no, line_no, word_no in page.get_text("words"):
            contents.append(
                {
                    "coordi": [
                        round(x0 / width, 3),
                        round(y0 / height, 3),
                        round(x1 / width, 3),
                        round(y1 / height, 3),
                    ],
                    "word": word,
                    "line_no": line_no,
                    "block_no": block_no,
                    "word_no": word_no,
                    "page_no": page_index,
                }
            )
    payload = {
        "zip_no": doc_no[:4],
        "doc_no": doc_no,
        "pdf_path": str(source_path),
        "img_size": image_size or [0, 0],
        "contents": contents,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return True


def preprocess_doc(benchmark, doc_id, *, pdf_dir, png_root, ocr_root, mode, dpi, max_pages, overwrite=False):
    import fitz

    source_path = pdf_path(benchmark, pdf_dir, doc_id)
    if not source_path.exists():
        raise FileNotFoundError(f"Missing {benchmark} PDF for doc_id={doc_id}: {source_path}")

    generated_pages = 0
    skipped_pages = 0
    generated_ocr = 0
    skipped_ocr = 0
    with fitz.open(source_path) as pdf:
        page_count = len(pdf) if max_pages is None else min(len(pdf), max_pages)
        for page_index in range(page_count):
            page = pdf[page_index]
            if mode in {"image", "both"}:
                output_path = png_page_path(benchmark, png_root, doc_id, page_index, dpi)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if page_needs_render(output_path, page, dpi, overwrite):
                    page.get_pixmap(dpi=dpi, alpha=False).save(output_path)
                    generated_pages += 1
                else:
                    skipped_pages += 1
            if benchmark == "mmlongbench" and mode in {"ocr", "both"}:
                if write_mmlongbench_ocr_page(ocr_root, doc_id, page_index, page, overwrite):
                    generated_ocr += 1
                else:
                    skipped_ocr += 1

        if benchmark == "longdocurl" and mode in {"ocr", "both"}:
            if write_longdocurl_ocr(ocr_root, doc_id, source_path, pdf, page_count, dpi, overwrite):
                generated_ocr = 1
            else:
                skipped_ocr = 1

    return {
        "doc_id": str(doc_id),
        "doc_key": document_key(benchmark, doc_id),
        "page_count": page_count,
        "generated_pages": generated_pages,
        "skipped_pages": skipped_pages,
        "generated_ocr": generated_ocr,
        "skipped_ocr": skipped_ocr,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.dpi < 1:
        raise ValueError("--dpi must be >= 1")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.max_pages is not None and args.max_pages < 1:
        raise ValueError("--max-pages must be >= 1")

    input_path, pdf_dir, png_root, ocr_root, max_pages = resolve_paths(args)
    doc_ids = load_doc_ids(args.benchmark, input_path, args.doc_id)
    if args.limit is not None:
        doc_ids = doc_ids[: args.limit]

    logger.info(
        "Preprocessing benchmark=%s documents=%s mode=%s dpi=%s max_pages=%s workers=%s",
        args.benchmark,
        len(doc_ids),
        args.mode,
        args.dpi,
        max_pages,
        args.workers,
    )
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                preprocess_doc,
                args.benchmark,
                doc_id,
                pdf_dir=pdf_dir,
                png_root=png_root,
                ocr_root=ocr_root,
                mode=args.mode,
                dpi=args.dpi,
                max_pages=max_pages,
                overwrite=args.overwrite,
            ): doc_id
            for doc_id in doc_ids
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            logger.info(
                "Processed doc_id=%s pages=%s images=%s/%s ocr=%s/%s",
                result["doc_id"],
                result["page_count"],
                result["generated_pages"],
                result["skipped_pages"],
                result["generated_ocr"],
                result["skipped_ocr"],
            )

    logger.info(
        "Finished benchmark=%s documents=%s generated_pages=%s skipped_pages=%s generated_ocr=%s skipped_ocr=%s",
        args.benchmark,
        len(results),
        sum(result["generated_pages"] for result in results),
        sum(result["skipped_pages"] for result in results),
        sum(result["generated_ocr"] for result in results),
        sum(result["skipped_ocr"] for result in results),
    )


if __name__ == "__main__":
    main()
