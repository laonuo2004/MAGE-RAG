#!/usr/bin/env python3
import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from tqdm import tqdm


CODE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_DIR))

from baselines.utils.benchmarks_related import (
    bgem3_doc_cache_variant,
    bgem3_query_cache_variant,
    build_token_chunks_from_pages,
    load_longdocurl_ocr_pages,
    load_longdocurl_vlm_text_pages,
)


DEFAULT_INPUT_PATH = CODE_DIR / "benchmarks" / "longdocurl" / "data" / "raw" / "LongDocURL.jsonl"
DEFAULT_DOC_OUTPUT_DIR = CODE_DIR / "benchmarks" / "longdocurl" / "tmp" / "bgem3" / "doc_embeddings"
DEFAULT_QUERY_OUTPUT_DIR = CODE_DIR / "benchmarks" / "longdocurl" / "tmp" / "bgem3" / "query_embeddings"
DEFAULT_METADATA_OUTPUT_DIR = CODE_DIR / "benchmarks" / "longdocurl" / "tmp" / "bgem3" / "chunk_metadata"
DEFAULT_CHECKPOINT = "/root/autodl-tmp/ylz/models/bge-m3"
DEFAULT_OCR_JSON_DIR = CODE_DIR / "benchmarks" / "longdocurl" / "data" / "pdf_jsons" / "4000-4999"
DEFAULT_IMAGE_PREFIX = CODE_DIR / "benchmarks" / "longdocurl" / "data" / "processed" / "pdf_pngs" / "4000-4999"
DEFAULT_MINERU_DIR = CODE_DIR / "benchmarks" / "longdocurl" / "data" / "pdfs_mineru" / "4000-4999"

logger = logging.getLogger("generate_longdocurl_bgem3_embeddings")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate BGEM3 dense chunk/query embeddings for LongDocURL.")
    parser.add_argument("--mode", choices=["doc", "query", "both"], default="both")
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--doc-output-dir", default=str(DEFAULT_DOC_OUTPUT_DIR))
    parser.add_argument("--query-output-dir", default=str(DEFAULT_QUERY_OUTPUT_DIR))
    parser.add_argument("--metadata-output-dir", default=str(DEFAULT_METADATA_OUTPUT_DIR))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer-name", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--devices", default="cuda:0")
    parser.add_argument("--doc-no", action="append", default=None)
    parser.add_argument("--question-id", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--chunk-overlap", type=int, default=20)
    parser.add_argument("--allow-cross-page", action="store_true", default=True)
    parser.add_argument("--no-allow-cross-page", action="store_false", dest="allow_cross_page")
    parser.add_argument("--max-cross-pages", type=int, default=None)
    parser.add_argument("--ocr-json-dir", default=str(DEFAULT_OCR_JSON_DIR))
    parser.add_argument("--image-prefix", default=str(DEFAULT_IMAGE_PREFIX))
    parser.add_argument("--mineru-dir", default=str(DEFAULT_MINERU_DIR))
    parser.add_argument("--mode-name", default="dense")
    parser.add_argument("--text-source", default="ocr")
    parser.add_argument("--use-fp16", action="store_true", default=True)
    parser.add_argument("--no-use-fp16", action="store_false", dest="use_fp16")
    return parser.parse_args()


def load_samples(input_path, image_prefix):
    samples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            sample = json.loads(line)
            sample.setdefault("question_id", idx)
            if image_prefix is not None:
                images = []
                for image_path in sample.get("images", []):
                    images.append(str(Path(image_prefix) / "/".join(str(image_path).split("/")[-2:])))
                sample["images"] = images
            samples.append(sample)
    return samples


def select_samples(samples, explicit_doc_nos, explicit_question_ids, limit):
    explicit_doc_nos = set(explicit_doc_nos or [])
    explicit_question_ids = set(explicit_question_ids or [])
    selected = []
    for sample in samples:
        if explicit_doc_nos and sample["doc_no"] not in explicit_doc_nos:
            continue
        if explicit_question_ids and sample["question_id"] not in explicit_question_ids:
            continue
        selected.append(sample)
    if limit is not None:
        selected = selected[:limit]
    return selected


def load_model(checkpoint, use_fp16, devices):
    from FlagEmbedding import BGEM3FlagModel

    return BGEM3FlagModel(checkpoint, use_fp16=use_fp16, devices=devices)


def tokenize_with_spans_factory(tokenizer_name):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    def tokenize_with_spans(text):
        encoded = tokenizer(
            "" if text is None else str(text),
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        spans = []
        for start, end in encoded["offset_mapping"]:
            if end <= start:
                continue
            spans.append({"start": int(start), "end": int(end)})
        return spans

    return tokenize_with_spans


def build_cfg(args):
    return {
        "benchmarks": {
            "name": "longdocurl",
            "ocr_json_dir": args.ocr_json_dir,
            "image_prefix": args.image_prefix,
            "mineru_dir": args.mineru_dir,
        }
    }


def encode_doc_chunks(model, chunks, output_path, metadata_path, overwrite, batch_size, max_length):
    if output_path.exists() and metadata_path.exists() and not overwrite:
        logger.info("Skipping existing BGEM3 doc embeddings: %s", output_path)
        return
    texts = [chunk["text"] for chunk in chunks]
    outputs = model.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    dense_vecs = torch.as_tensor(outputs["dense_vecs"], dtype=torch.float32, device="cpu")
    save_file({"dense_vecs": dense_vecs}, output_path)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info("Saved %s and %s", output_path, metadata_path)


def encode_query(model, question, output_path, overwrite, batch_size, max_length):
    if output_path.exists() and not overwrite:
        logger.info("Skipping existing BGEM3 query embeddings: %s", output_path)
        return
    outputs = model.encode(
        [question],
        batch_size=batch_size,
        max_length=max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    query_dense = torch.as_tensor(outputs["dense_vecs"][0], dtype=torch.float32, device="cpu")
    save_file({"query_dense_vec": query_dense}, output_path)
    logger.info("Saved %s", output_path)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    samples = load_samples(args.input_path, args.image_prefix)
    selected = select_samples(samples, args.doc_no, args.question_id, args.limit)

    doc_variant = bgem3_doc_cache_variant(
        args.checkpoint,
        args.mode_name,
        args.text_source,
        args.chunk_size,
        args.chunk_overlap,
        args.allow_cross_page,
        args.max_cross_pages,
    )
    query_variant = bgem3_query_cache_variant(args.checkpoint, args.mode_name)
    doc_output_dir = Path(args.doc_output_dir) / doc_variant
    query_output_dir = Path(args.query_output_dir) / query_variant
    metadata_output_dir = Path(args.metadata_output_dir) / doc_variant
    doc_output_dir.mkdir(parents=True, exist_ok=True)
    query_output_dir.mkdir(parents=True, exist_ok=True)
    metadata_output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, args.use_fp16, args.devices)
    tokenize_with_spans = tokenize_with_spans_factory(args.tokenizer_name)
    cfg = build_cfg(args)

    for sample in tqdm(selected, desc="LongDocURL BGEM3 embeddings"):
        if args.mode in {"doc", "both"}:
            if args.text_source == "ocr":
                pages, _ = load_longdocurl_ocr_pages(sample, cfg["benchmarks"])
            elif args.text_source == "vlm_text":
                pages, _ = load_longdocurl_vlm_text_pages(sample, cfg["benchmarks"])
            else:
                raise ValueError(f"Unsupported text_source for LongDocURL BGEM3: {args.text_source}")
            chunks = build_token_chunks_from_pages(
                pages,
                tokenize_with_spans,
                args.chunk_size,
                args.chunk_overlap,
                allow_cross_page=args.allow_cross_page,
                max_cross_pages=args.max_cross_pages,
            )
            for idx, chunk in enumerate(chunks):
                chunk["chunk_id"] = idx
            encode_doc_chunks(
                model,
                chunks,
                doc_output_dir / f"{sample['question_id']}.safetensors",
                metadata_output_dir / f"{sample['question_id']}.json",
                args.overwrite,
                args.batch_size,
                args.max_length,
            )

        if args.mode in {"query", "both"}:
            encode_query(
                model,
                sample["question"],
                query_output_dir / f"{sample['question_id']}.safetensors",
                args.overwrite,
                args.batch_size,
                args.max_length,
            )


if __name__ == "__main__":
    main()
