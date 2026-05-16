import json
import logging
from pathlib import Path

import torch
from safetensors.torch import save_file

from baselines.colbertv2 import ColBERTv2ContextBuilder
from baselines.utils.benchmarks_related import (
    build_token_chunks_from_pages,
    colbertv2_doc_cache_variant,
    colbertv2_query_cache_variant,
    load_longdocurl_ocr_pages,
    load_mmlongbench_ocr_pages,
)
from utils.config_utils import get_config_value, require_config_value

logger = logging.getLogger(__name__)


def prepare_colbertv2_cache(cfg):
    if require_config_value(cfg, "baselines.name") != "colbertv2":
        return

    benchmark_name = require_config_value(cfg, "benchmarks.name")
    builder = ColBERTv2ContextBuilder(cfg)
    samples = _load_samples(cfg, benchmark_name)
    if not samples:
        return

    checkpoint = _load_checkpoint(require_config_value(cfg, "baselines.checkpoint"))
    tokenize_with_spans = _tokenize_with_spans_factory(checkpoint)
    doc_variant = colbertv2_doc_cache_variant(
        require_config_value(cfg, "baselines.checkpoint"),
        builder.chunk_size,
        builder.chunk_overlap,
        builder.allow_cross_page,
        builder.max_cross_pages,
    )
    query_variant = colbertv2_query_cache_variant(require_config_value(cfg, "baselines.checkpoint"))

    if benchmark_name == "mmlongbench":
        sample_by_doc = {}
        for sample in samples:
            sample_by_doc.setdefault(sample["doc_id"], sample)
        for doc_id, sample in sample_by_doc.items():
            doc_key = builder._mmlongbench_doc_key(doc_id)
            doc_output_path = _variant_root(cfg, "baselines.doc_embeddings_colbertv2", benchmark_name, doc_variant) / f"{doc_key}.safetensors"
            metadata_output_path = _variant_root(cfg, "baselines.chunk_metadata_colbertv2", benchmark_name, doc_variant) / f"{doc_key}.json"
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
                _encode_doc_chunks(checkpoint, chunks, doc_output_path, metadata_output_path)

        for sample in samples:
            query_key = str(sample["question_id"])
            query_output_path = _variant_root(cfg, "baselines.query_embeddings_colbertv2", benchmark_name, query_variant) / f"{query_key}.safetensors"
            if not query_output_path.exists():
                _encode_query(checkpoint, sample["question"], query_output_path)
        return

    if benchmark_name == "longdocurl":
        for sample in samples:
            doc_key = str(sample["question_id"])
            query_key = str(sample["question_id"])
            doc_output_path = _variant_root(cfg, "baselines.doc_embeddings_colbertv2", benchmark_name, doc_variant) / f"{doc_key}.safetensors"
            metadata_output_path = _variant_root(cfg, "baselines.chunk_metadata_colbertv2", benchmark_name, doc_variant) / f"{doc_key}.json"
            query_output_path = _variant_root(cfg, "baselines.query_embeddings_colbertv2", benchmark_name, query_variant) / f"{query_key}.safetensors"
            if not doc_output_path.exists() or not metadata_output_path.exists():
                pages, _ = load_longdocurl_ocr_pages(sample, require_config_value(cfg, "benchmarks"))
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
                _encode_doc_chunks(checkpoint, chunks, doc_output_path, metadata_output_path)
            if not query_output_path.exists():
                _encode_query(checkpoint, sample["question"], query_output_path)
        return

    raise ValueError(f"Unsupported benchmark for ColBERTv2 cache preparation: {benchmark_name}")


def _variant_root(cfg, field, benchmark_name, variant):
    root = require_config_value(cfg, field)
    value = root if isinstance(root, str) else get_config_value(root, benchmark_name)
    if not value:
        raise ValueError(f"Missing {field}.{benchmark_name}")
    path = Path(str(value)) / variant
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_checkpoint(checkpoint_path):
    from colbert.infra import ColBERTConfig
    from colbert.modeling.checkpoint import Checkpoint
    from transformers.modeling_utils import PreTrainedModel

    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}

    config = ColBERTConfig(checkpoint=checkpoint_path)
    return Checkpoint(checkpoint_path, colbert_config=config)


def _tokenize_with_spans_factory(checkpoint):
    tokenizer = checkpoint.doc_tokenizer.tok

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


def _encode_doc_chunks(checkpoint, chunks, output_path, metadata_path):
    texts = [chunk["text"] for chunk in chunks]
    doc_embs = checkpoint.docFromText(texts, bsize=8, keep_dims="flatten", to_cpu=False)
    if not isinstance(doc_embs, tuple) or len(doc_embs) < 2:
        raise ValueError("Unexpected ColBERTv2 docFromText return value.")
    embs, doclens = doc_embs[:2]
    embs = embs.detach().to(dtype=torch.float32, device="cpu")
    doclens = torch.as_tensor(doclens, dtype=torch.int32, device="cpu")
    save_file({"embeddings": embs, "doclens": doclens}, output_path)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info("Saved %s and %s", output_path, metadata_path)


def _encode_query(checkpoint, question, output_path):
    query_emb = checkpoint.queryFromText([question], bsize=8, to_cpu=False)
    if isinstance(query_emb, (list, tuple)):
        query_emb = query_emb[0]
    if isinstance(query_emb, torch.Tensor):
        if query_emb.ndim == 3 and query_emb.shape[0] == 1:
            query_emb = query_emb[0]
    else:
        query_emb = torch.as_tensor(query_emb)
    query_emb = query_emb.detach().to(dtype=torch.float32, device="cpu")
    if query_emb.ndim != 2:
        raise ValueError(f"Unexpected ColBERTv2 query embedding shape: {tuple(query_emb.shape)}")
    save_file({"query_embedding": query_emb}, output_path)
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
