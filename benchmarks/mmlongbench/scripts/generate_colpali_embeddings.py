#!/usr/bin/env python3
import argparse
import base64
import json
import logging
import re
from pathlib import Path

import requests
import torch
from safetensors.torch import save_file
from tqdm import tqdm


CODE_DIR = Path(__file__).resolve().parents[3]

import sys
sys.path.insert(0, str(CODE_DIR))

from benchmarks.utils.data_utils import mmlongbench_file_id


DEFAULT_INPUT_PATH = CODE_DIR / "benchmarks" / "mmlongbench" / "data" / "raw" / "samples.json"
DEFAULT_IMAGE_ROOT = CODE_DIR / "benchmarks" / "mmlongbench" / "data" / "processed" / "pdf_pngs"
DEFAULT_OUTPUT_DIR = CODE_DIR / "benchmarks" / "mmlongbench" / "data" / "cache" / "colpali" / "pdf_embeddings"
DEFAULT_QUESTION_OUTPUT_DIR = CODE_DIR / "benchmarks" / "mmlongbench" / "data" / "cache" / "colpali" / "question_embeddings"
DEFAULT_VLLM_URL = "http://127.0.0.1:8020"
DEFAULT_MODEL_NAME = "colpali-v1.3"

logger = logging.getLogger("generate_mmlongbench_colpali_embeddings")
PAGE_RE_TEMPLATE = r"^page_(?P<page_num>\d{{4}})_dpi{dpi}\.png$"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ColPali PDF and question embedding safetensors for MMLongBench."
    )
    parser.add_argument("--mode", choices=["pdf", "question", "both"], default="both", help="Embedding type to generate.")
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH), help="MMLongBench samples.json path.")
    parser.add_argument("--image-root", default=str(DEFAULT_IMAGE_ROOT), help="Root PNG cache directory.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for .safetensors outputs.")
    parser.add_argument(
        "--question-output-dir",
        default=str(DEFAULT_QUESTION_OUTPUT_DIR),
        help="Directory for question .safetensors outputs.",
    )
    parser.add_argument("--vllm-url", default=DEFAULT_VLLM_URL, help="Base URL for the vLLM ColPali service.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Served vLLM model name.")
    parser.add_argument("--request-timeout", type=float, default=180.0, help="HTTP request timeout in seconds.")
    parser.add_argument("--batch-size", type=int, default=4, help="Question encoding batch size; PDF pages are encoded one request per page.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N selected documents/questions.")
    parser.add_argument("--doc-id", action="append", default=None, help="Specific doc_id to process. Can be repeated.")
    parser.add_argument(
        "--question-id",
        action="append",
        default=None,
        help="Specific question_id to process. Can be repeated.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate embeddings that already exist.")
    parser.add_argument("--dpi", type=int, default=144, help="DPI suffix to select cached page PNGs.")
    return parser.parse_args()


def load_doc_ids(input_path, explicit_doc_ids):
    if explicit_doc_ids:
        return sorted(set(explicit_doc_ids))

    with open(input_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    return sorted({sample["doc_id"] for sample in samples})


def load_question_samples(input_path, explicit_doc_ids, explicit_question_ids):
    explicit_doc_ids = set(explicit_doc_ids or [])
    explicit_question_ids = set(explicit_question_ids or [])
    samples_by_id = {}

    with open(input_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    for sample in samples:
        question_id = sample["question_id"]
        doc_id = sample["doc_id"]
        if explicit_doc_ids and doc_id not in explicit_doc_ids:
            continue
        if explicit_question_ids and question_id not in explicit_question_ids:
            continue
        samples_by_id.setdefault(
            question_id,
            {
                "question_id": question_id,
                "doc_id": doc_id,
                "question": sample["question"],
            },
        )

    return [samples_by_id[question_id] for question_id in sorted(samples_by_id)]


def find_page_images(image_root, doc_id, dpi):
    file_id = mmlongbench_file_id(doc_id)
    page_dir = image_root / file_id
    if not page_dir.exists():
        raise FileNotFoundError(f"Missing PNG directory for doc_id={doc_id}, file_id={file_id}: {page_dir}")

    page_re = re.compile(PAGE_RE_TEMPLATE.format(dpi=re.escape(str(dpi))))
    pages = []
    for path in page_dir.glob(f"page_*_dpi{dpi}.png"):
        match = page_re.match(path.name)
        if not match:
            continue
        pages.append((int(match.group("page_num")), path))

    if not pages:
        raise FileNotFoundError(f"No dpi={dpi} PNG pages found for doc_id={doc_id}, file_id={file_id} in {page_dir}")

    pages.sort(key=lambda item: item[0])
    page_numbers = [num for num, _ in pages]
    expected = list(range(1, page_numbers[-1] + 1))
    if page_numbers != expected:
        missing = sorted(set(expected) - set(page_numbers))
        raise ValueError(f"Non-contiguous pages for doc_id={doc_id}; missing 1-based page numbers: {missing}")
    return file_id, pages


def write_manifest_record(manifest_path, record):
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _vllm_endpoint(args, route):
    return f"{args.vllm_url.rstrip('/')}/{route.lstrip('/')}"


def _post_pooling(args, payload):
    response = requests.post(_vllm_endpoint(args, "pooling"), json=payload, timeout=args.request_timeout)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"vLLM /pooling request failed: {response.text[:1000]}") from exc
    return response.json()


def _pooling_item_to_tensor(item):
    if "data" not in item:
        raise KeyError(f'vLLM pooling response item missing "data": keys={list(item)}')
    tensor = torch.tensor(item["data"], dtype=torch.float32)
    if tensor.ndim != 2:
        raise ValueError(f"Expected token embeddings [tokens, dim], got {tuple(tensor.shape)}")
    return tensor.to(dtype=torch.bfloat16, device="cpu")


def encode_image_path(args, image_path):
    image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    payload = {
        "model": args.model,
        "task": "token_embed",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<image>"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            }
        ],
    }
    data = _post_pooling(args, payload)
    return _pooling_item_to_tensor(data["data"][0])


def encode_queries(args, queries):
    payload = {
        "model": args.model,
        "task": "token_embed",
        "input": queries,
    }
    data = _post_pooling(args, payload)
    return [_pooling_item_to_tensor(item) for item in data["data"]]


def encode_pdfs(args, doc_ids, doc_pages, output_dir):
    manifest_path = output_dir / "manifest.jsonl"
    for doc_id in tqdm(doc_ids, desc="MMLongBench ColPali PDF embeddings"):
        file_id, pages = doc_pages[doc_id]
        output_path = output_dir / f"{file_id}.safetensors"
        page_numbers = [num for num, _ in pages]
        page_indices = [num - 1 for num in page_numbers]
        image_paths = [str(path) for _, path in pages]
        status = "skipped"

        if output_path.exists() and not args.overwrite:
            logger.info("Skipping existing PDF embedding: %s", output_path)
        else:
            doc_embs = [encode_image_path(args, path) for _, path in tqdm(pages, desc=f"{doc_id} pages", leave=False)]
            token_shapes = {tuple(emb.shape) for emb in doc_embs}
            if len(token_shapes) != 1:
                raise ValueError(f"Cannot stack variable page embedding shapes for doc_id={doc_id}: {sorted(token_shapes)}")
            doc_embs = torch.stack(doc_embs, dim=0).to(torch.bfloat16)
            save_file({"embeddings": doc_embs}, output_path)
            status = "generated"
            logger.info("Saved %s with shape %s", output_path, tuple(doc_embs.shape))

        write_manifest_record(
            manifest_path,
            {
                "doc_id": doc_id,
                "file_id": file_id,
                "embedding_path": str(output_path),
                "page_count": len(page_numbers),
                "page_indices": page_indices,
                "page_numbers": page_numbers,
                "image_paths": image_paths,
                "model": args.model,
                "vllm_url": args.vllm_url,
                "dtype": "bfloat16",
                "status": status,
            },
        )


def _chunks(items, size):
    if size <= 0:
        raise ValueError("--batch-size must be > 0")
    for index in range(0, len(items), size):
        yield items[index:index + size]


def encode_questions(args, question_samples, output_dir):
    manifest_path = output_dir / "manifest.jsonl"
    pending_samples = [
        sample
        for sample in question_samples
        if args.overwrite or not (output_dir / f"{sample['question_id']}.safetensors").exists()
    ]
    pending_embeddings = {}
    for batch in tqdm(list(_chunks(pending_samples, args.batch_size)), desc="ColPali question batches", leave=False):
        embeddings = encode_queries(args, [sample["question"] for sample in batch])
        if len(embeddings) != len(batch):
            raise ValueError(f"vLLM returned {len(embeddings)} embeddings for {len(batch)} queries")
        for sample, embedding in zip(batch, embeddings):
            pending_embeddings[sample["question_id"]] = embedding

    for sample in tqdm(question_samples, desc="MMLongBench ColPali question embeddings"):
        question_id = sample["question_id"]
        output_path = output_dir / f"{question_id}.safetensors"
        status = "skipped"

        if output_path.exists() and not args.overwrite:
            logger.info("Skipping existing question embedding: %s", output_path)
        else:
            query_emb = pending_embeddings[question_id]
            save_file({"query_embedding": query_emb}, output_path)
            status = "generated"
            logger.info("Saved %s with shape %s", output_path, tuple(query_emb.shape))

        write_manifest_record(
            manifest_path,
            {
                "question_id": question_id,
                "doc_id": sample["doc_id"],
                "question": sample["question"],
                "embedding_path": str(output_path),
                "model": args.model,
                "vllm_url": args.vllm_url,
                "dtype": "bfloat16",
                "status": status,
            },
        )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    output_dir = Path(args.output_dir)
    question_output_dir = Path(args.question_output_dir)
    pdf_enabled = args.mode in {"pdf", "both"}
    question_enabled = args.mode in {"question", "both"}

    doc_ids = []
    doc_pages = {}
    pdf_pending = []
    if pdf_enabled:
        image_root = Path(args.image_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        doc_ids = load_doc_ids(args.input_path, args.doc_id)
        if args.limit is not None:
            doc_ids = doc_ids[: args.limit]
        logger.info("Selected %s MMLongBench documents.", len(doc_ids))
        doc_pages = {doc_id: find_page_images(image_root, doc_id, args.dpi) for doc_id in doc_ids}
        for doc_id in doc_ids:
            file_id, _ = doc_pages[doc_id]
            if args.overwrite or not (output_dir / f"{file_id}.safetensors").exists():
                pdf_pending.append(doc_id)

    question_samples = []
    question_pending = []
    if question_enabled:
        question_output_dir.mkdir(parents=True, exist_ok=True)
        question_samples = load_question_samples(args.input_path, args.doc_id, args.question_id)
        if args.limit is not None:
            question_samples = question_samples[: args.limit]
        logger.info("Selected %s MMLongBench questions.", len(question_samples))
        question_pending = [
            sample["question_id"]
            for sample in question_samples
            if args.overwrite or not (question_output_dir / f"{sample['question_id']}.safetensors").exists()
        ]

    if not (pdf_pending or question_pending):
        logger.info("All selected embeddings already exist; writing skip records only.")
    else:
        logger.info("Using vLLM ColPali service at %s with model=%s", args.vllm_url, args.model)

    if pdf_enabled:
        encode_pdfs(args, doc_ids, doc_pages, output_dir)
    if question_enabled:
        encode_questions(args, question_samples, question_output_dir)


if __name__ == "__main__":
    main()
