import json
import logging
from pathlib import Path

import torch
from safetensors.torch import save_file

from baselines.bgem3 import BGEM3ContextBuilder
from baselines.utils.benchmarks_related import (
    bgem3_doc_cache_variant,
    bgem3_query_cache_variant,
    build_token_chunks_from_pages,
    load_longdocurl_ocr_pages,
    load_longdocurl_vlm_text_pages,
    load_mmlongbench_ocr_pages,
)
from utils.config_utils import get_config_value, require_config_value

logger = logging.getLogger(__name__)


def prepare_bgem3_cache(cfg):
    if require_config_value(cfg, "baselines.name") != "bgem3":
        return

    benchmark_name = require_config_value(cfg, "benchmarks.name")
    builder = BGEM3ContextBuilder(cfg)
    samples = _load_samples(cfg, benchmark_name)
    if not samples:
        return

    model = _load_model(cfg)
    batch_size = int(get_config_value(cfg, "baselines.batch_size", 8))
    max_length = int(get_config_value(cfg, "baselines.max_length", 8192))
    tokenize_with_spans = _tokenize_with_spans_factory(cfg)
    doc_variant = bgem3_doc_cache_variant(
        require_config_value(cfg, "baselines.checkpoint"),
        builder.mode,
        builder.text_source,
        builder.chunk_size,
        builder.chunk_overlap,
        builder.allow_cross_page,
        builder.max_cross_pages,
    )
    query_variant = bgem3_query_cache_variant(require_config_value(cfg, "baselines.checkpoint"), builder.mode)

    if benchmark_name == "mmlongbench":
        if builder.text_source != "ocr":
            raise ValueError("MMLongBench BGEM3 currently only supports text_source=ocr.")
        sample_by_doc = {}
        for sample in samples:
            sample_by_doc.setdefault(sample["doc_id"], sample)
        for doc_id, sample in sample_by_doc.items():
            doc_key = builder._mmlongbench_doc_key(doc_id)
            doc_output_path = _variant_root(cfg, "baselines.doc_embeddings_bgem3", benchmark_name, doc_variant) / f"{doc_key}.safetensors"
            metadata_output_path = _variant_root(cfg, "baselines.chunk_metadata_bgem3", benchmark_name, doc_variant) / f"{doc_key}.json"
            if not doc_output_path.exists() or not metadata_output_path.exists():
                pages, _ = load_mmlongbench_ocr_pages(sample, require_config_value(cfg, "benchmarks"))
                chunks = build_token_chunks_from_pages(
                    pages,
                    tokenize_with_spans,
                    builder.chunk_size,
                    builder.chunk_overlap,
                    allow_cross_page=builder.allow_cross_page,
                    max_cross_pages=builder.max_cross_pages,
                )
                for idx, chunk in enumerate(chunks):
                    chunk["chunk_id"] = idx
                _encode_doc_chunks(model, chunks, doc_output_path, metadata_output_path, batch_size, max_length)

        for sample in samples:
            query_key = str(sample["question_id"])
            query_output_path = _variant_root(cfg, "baselines.query_embeddings_bgem3", benchmark_name, query_variant) / f"{query_key}.safetensors"
            if not query_output_path.exists():
                _encode_query(model, sample["question"], query_output_path, batch_size, max_length)
        return

    if benchmark_name == "longdocurl":
        for sample in samples:
            doc_key = str(sample["question_id"])
            query_key = str(sample["question_id"])
            doc_output_path = _variant_root(cfg, "baselines.doc_embeddings_bgem3", benchmark_name, doc_variant) / f"{doc_key}.safetensors"
            metadata_output_path = _variant_root(cfg, "baselines.chunk_metadata_bgem3", benchmark_name, doc_variant) / f"{doc_key}.json"
            query_output_path = _variant_root(cfg, "baselines.query_embeddings_bgem3", benchmark_name, query_variant) / f"{query_key}.safetensors"
            if not doc_output_path.exists() or not metadata_output_path.exists():
                if builder.text_source == "ocr":
                    pages, _ = load_longdocurl_ocr_pages(sample, require_config_value(cfg, "benchmarks"))
                elif builder.text_source == "vlm_text":
                    pages, _ = load_longdocurl_vlm_text_pages(sample, require_config_value(cfg, "benchmarks"))
                else:
                    raise ValueError(f"Unsupported BGEM3 text_source: {builder.text_source}")
                chunks = build_token_chunks_from_pages(
                    pages,
                    tokenize_with_spans,
                    builder.chunk_size,
                    builder.chunk_overlap,
                    allow_cross_page=builder.allow_cross_page,
                    max_cross_pages=builder.max_cross_pages,
                )
                for idx, chunk in enumerate(chunks):
                    chunk["chunk_id"] = idx
                _encode_doc_chunks(model, chunks, doc_output_path, metadata_output_path, batch_size, max_length)
            if not query_output_path.exists():
                _encode_query(model, sample["question"], query_output_path, batch_size, max_length)
        return

    raise ValueError(f"Unsupported benchmark for BGEM3 cache preparation: {benchmark_name}")


def _variant_root(cfg, field, benchmark_name, variant):
    root = require_config_value(cfg, field)
    value = root if isinstance(root, str) else get_config_value(root, benchmark_name)
    if not value:
        raise ValueError(f"Missing {field}.{benchmark_name}")
    path = Path(str(value)) / variant
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_model(cfg):
    from FlagEmbedding import BGEM3FlagModel

    checkpoint = require_config_value(cfg, "baselines.checkpoint")
    use_fp16 = bool(get_config_value(cfg, "baselines.use_fp16", True))
    devices = get_config_value(cfg, "baselines.devices", "cuda:0")
    return BGEM3FlagModel(checkpoint, use_fp16=use_fp16, devices=devices)


def _tokenize_with_spans_factory(cfg):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(require_config_value(cfg, "baselines.tokenizer_name"), use_fast=True)

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


def _encode_doc_chunks(model, chunks, output_path, metadata_path, batch_size, max_length):
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


def _encode_query(model, question, output_path, batch_size, max_length):
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


def _load_samples(cfg, benchmark_name):
    if benchmark_name == "mmlongbench":
        input_path = require_config_value(cfg, "benchmarks.input_path")
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)

    qa_file = require_config_value(cfg, "benchmarks.qa_file")
    image_prefix = get_config_value(cfg, "benchmarks.image_prefix")
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
                    images.append(str(Path(image_prefix) / "/".join(str(image_path).split("/")[-2:])))
                sample["images"] = images
            samples.append(sample)
    return samples
