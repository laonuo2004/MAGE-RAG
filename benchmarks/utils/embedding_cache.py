import json
import logging
from pathlib import Path

from benchmarks.utils.data_utils import (
    bgem3_cache_root,
    colbertv2_cache_root,
    load_longdocurl_samples,
    load_mmlongbench_samples,
    mmlongbench_file_id,
)
from benchmarks.utils.document_preprocess import (
    bgem3_doc_cache_variant,
    bgem3_query_cache_variant,
    build_token_chunks_from_pages,
    colbertv2_doc_cache_variant,
    colbertv2_query_cache_variant,
    load_longdocurl_ocr_pages,
    load_longdocurl_vlm_text_pages,
    load_mmlongbench_ocr_pages,
)
from utils.config_utils import get_config_value, require_config_value

logger = logging.getLogger(__name__)


def prepare_bgem3_cache(cfg):
    if require_config_value(cfg, "baselines.name") != "bgem3":
        return
    generate_embedding_cache(cfg, baseline_name="bgem3")


def prepare_colbertv2_cache(cfg):
    if require_config_value(cfg, "baselines.name") != "colbertv2":
        return
    generate_embedding_cache(cfg, baseline_name="colbertv2")


def prepare_embedding_cache(cfg):
    baseline_name = require_config_value(cfg, "baselines.name")
    if baseline_name in {"bgem3", "colbertv2"}:
        generate_embedding_cache(cfg, baseline_name=baseline_name)


def generate_embedding_cache(
    cfg,
    *,
    baseline_name=None,
    mode="both",
    overwrite=False,
    doc_ids=None,
    question_ids=None,
):
    baseline_name = baseline_name or require_config_value(cfg, "baselines.name")
    if baseline_name not in {"bgem3", "colbertv2"}:
        return
    if mode not in {"doc", "query", "both"}:
        raise ValueError(f"Unsupported embedding cache mode: {mode}")

    params = _EmbeddingCacheParams.from_cfg(cfg, baseline_name=baseline_name)
    params.samples = _select_samples(params, doc_ids=doc_ids, question_ids=question_ids)
    if not params.samples:
        return

    if baseline_name == "bgem3":
        model = _load_bgem3_model(cfg)
        tokenize_with_spans = _bgem3_tokenize_with_spans_factory(cfg)
        for record in _iter_cache_records(params, tokenize_with_spans):
            if mode in {"doc", "both"} and (overwrite or not record.doc_ready):
                _encode_bgem3_doc_chunks(
                    model,
                    record.get_chunks(),
                    record.doc_embedding_path,
                    record.metadata_path,
                    params.batch_size,
                    params.max_length,
                )
            if mode in {"query", "both"} and (overwrite or not record.query_ready):
                _encode_bgem3_query(
                    model,
                    record.question,
                    record.query_embedding_path,
                    params.batch_size,
                    params.max_length,
                )
        return

    checkpoint = _load_colbertv2_checkpoint(params.checkpoint)
    tokenize_with_spans = _colbertv2_tokenize_with_spans_factory(checkpoint)
    for record in _iter_cache_records(params, tokenize_with_spans):
        if mode in {"doc", "both"} and (overwrite or not record.doc_ready):
            _encode_colbertv2_doc_chunks(
                checkpoint,
                record.get_chunks(),
                record.doc_embedding_path,
                record.metadata_path,
                params.batch_size,
            )
        if mode in {"query", "both"} and (overwrite or not record.query_ready):
            _encode_colbertv2_query(checkpoint, record.question, record.query_embedding_path, params.batch_size)


def _select_samples(params, *, doc_ids=None, question_ids=None):
    doc_ids = {str(value) for value in (doc_ids or [])}
    question_ids = {str(value) for value in (question_ids or [])}
    if not doc_ids and not question_ids:
        return params.samples
    selected = []
    for sample in params.samples:
        if params.benchmark_name == "mmlongbench":
            doc_keys = {str(sample.get("doc_id")), mmlongbench_file_id(sample.get("doc_id"))}
        else:
            doc_keys = {str(sample.get("doc_no"))}
        if doc_ids and not (doc_keys & doc_ids):
            continue
        if question_ids and str(sample.get("question_id")) not in question_ids:
            continue
        selected.append(sample)
    return selected


class _EmbeddingCacheParams:
    def __init__(
        self,
        *,
        baseline_name,
        benchmark_name,
        checkpoint,
        mode,
        text_source,
        chunk_size,
        chunk_overlap,
        allow_cross_page,
        max_cross_pages,
        batch_size,
        max_length,
        samples,
        benchmark_cfg,
        output_dirs,
    ):
        self.baseline_name = baseline_name
        self.benchmark_name = benchmark_name
        self.checkpoint = checkpoint
        self.mode = mode
        self.text_source = text_source
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.allow_cross_page = allow_cross_page
        self.max_cross_pages = max_cross_pages
        self.batch_size = batch_size
        self.max_length = max_length
        self.samples = samples
        self.benchmark_cfg = benchmark_cfg
        self.output_dirs = output_dirs
        self.doc_variant = self._doc_variant()
        self.query_variant = self._query_variant()

    @classmethod
    def from_cfg(cls, cfg, baseline_name=None):
        benchmark_name = require_config_value(cfg, "benchmarks.name")
        baseline_name = baseline_name or require_config_value(cfg, "baselines.name")
        max_cross_pages = get_config_value(cfg, "baselines.params.max_cross_pages", None)
        if max_cross_pages is not None:
            max_cross_pages = int(max_cross_pages)
        return cls(
            baseline_name=baseline_name,
            benchmark_name=benchmark_name,
            checkpoint=require_config_value(cfg, "baselines.checkpoint"),
            mode=str(get_config_value(cfg, "baselines.params.mode", "dense")),
            text_source=str(get_config_value(cfg, "baselines.params.text_source", "ocr")),
            chunk_size=int(get_config_value(cfg, "baselines.params.chunk_size", 200)),
            chunk_overlap=int(get_config_value(cfg, "baselines.params.chunk_overlap", 20)),
            allow_cross_page=bool(get_config_value(cfg, "baselines.params.allow_cross_page", True)),
            max_cross_pages=max_cross_pages,
            batch_size=int(get_config_value(cfg, "baselines.batch_size", 8)),
            max_length=int(get_config_value(cfg, "baselines.max_length", 8192)),
            samples=_limit_samples(_load_samples(cfg, benchmark_name), get_config_value(cfg, "cache_sample_limit", None)),
            benchmark_cfg=require_config_value(cfg, "benchmarks"),
            output_dirs=get_config_value(cfg, "cache_output_dirs", {}) or {},
        )

    def cache_root(self, cache_kind):
        configured = get_config_value(self.output_dirs, cache_kind)
        if configured:
            return Path(configured)
        if self.baseline_name == "bgem3":
            return bgem3_cache_root(self.benchmark_name, cache_kind)
        if self.baseline_name == "colbertv2":
            return colbertv2_cache_root(self.benchmark_name, cache_kind)
        raise ValueError(f"Unsupported embedding cache baseline: {self.baseline_name}")

    def _doc_variant(self):
        if self.baseline_name == "bgem3":
            return bgem3_doc_cache_variant(
                self.checkpoint,
                self.mode,
                self.text_source,
                self.chunk_size,
                self.chunk_overlap,
                self.allow_cross_page,
                self.max_cross_pages,
            )
        return colbertv2_doc_cache_variant(
            self.checkpoint,
            self.chunk_size,
            self.chunk_overlap,
            self.allow_cross_page,
            self.max_cross_pages,
        )

    def _query_variant(self):
        if self.baseline_name == "bgem3":
            return bgem3_query_cache_variant(self.checkpoint, self.mode)
        return colbertv2_query_cache_variant(self.checkpoint)


class _CacheRecord:
    def __init__(self, *, chunks_factory, question, doc_embedding_path, metadata_path, query_embedding_path):
        self._chunks_factory = chunks_factory
        self._chunks = None
        self.question = question
        self.doc_embedding_path = doc_embedding_path
        self.metadata_path = metadata_path
        self.query_embedding_path = query_embedding_path

    @property
    def doc_ready(self):
        return self.doc_embedding_path.exists() and self.metadata_path.exists()

    @property
    def query_ready(self):
        return self.query_embedding_path.exists()

    def get_chunks(self):
        if self._chunks is None:
            self._chunks = self._chunks_factory()
        return self._chunks


def _iter_cache_records(params, tokenize_with_spans):
    if params.benchmark_name == "mmlongbench":
        if params.text_source != "ocr":
            raise ValueError(f"MMLongBench {params.baseline_name} cache currently only supports text_source=ocr.")
        sample_by_doc = {}
        for sample in params.samples:
            sample_by_doc.setdefault(sample["doc_id"], sample)
        for sample in params.samples:
            doc_key = mmlongbench_file_id(sample["doc_id"])
            doc_sample = sample_by_doc[sample["doc_id"]]
            yield _record_for_sample(
                params,
                doc_key,
                str(sample["question_id"]),
                sample["question"],
                lambda doc_sample=doc_sample: _build_mmlongbench_chunks(params, doc_sample, tokenize_with_spans),
            )
        return

    if params.benchmark_name == "longdocurl":
        for sample in params.samples:
            doc_key = str(sample["question_id"])
            query_key = str(sample["question_id"])
            yield _record_for_sample(
                params,
                doc_key,
                query_key,
                sample["question"],
                lambda sample=sample: _build_longdocurl_chunks(params, sample, tokenize_with_spans),
            )
        return

    raise ValueError(f"Unsupported benchmark for embedding cache preparation: {params.benchmark_name}")


def _record_for_sample(params, doc_key, query_key, question, chunks_factory):
    return _CacheRecord(
        chunks_factory=chunks_factory,
        question=question,
        doc_embedding_path=_variant_root(params.cache_root("doc_embeddings"), params.doc_variant) / f"{doc_key}.safetensors",
        metadata_path=_variant_root(params.cache_root("chunk_metadata"), params.doc_variant) / f"{doc_key}.json",
        query_embedding_path=_variant_root(params.cache_root("query_embeddings"), params.query_variant) / f"{query_key}.safetensors",
    )


def _build_mmlongbench_chunks(params, sample, tokenize_with_spans):
    pages, _ = load_mmlongbench_ocr_pages(sample, params.benchmark_cfg)
    return _build_chunks(params, pages, tokenize_with_spans)


def _build_longdocurl_chunks(params, sample, tokenize_with_spans):
    pages, _ = _load_longdocurl_pages(sample, params)
    return _build_chunks(params, pages, tokenize_with_spans)


def _build_chunks(params, pages, tokenize_with_spans):
    chunks = build_token_chunks_from_pages(
        pages,
        tokenize_with_spans,
        params.chunk_size,
        params.chunk_overlap,
        allow_cross_page=params.allow_cross_page,
        max_cross_pages=params.max_cross_pages,
    )
    for idx, chunk in enumerate(chunks):
        chunk["chunk_id"] = idx
    return chunks


def _load_longdocurl_pages(sample, params):
    if params.text_source == "ocr":
        return load_longdocurl_ocr_pages(sample, params.benchmark_cfg)
    if params.text_source == "vlm_text":
        return load_longdocurl_vlm_text_pages(sample, params.benchmark_cfg)
    raise ValueError(f"Unsupported {params.baseline_name} text_source: {params.text_source}")


def _variant_root(root, variant):
    path = Path(root) / variant
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_bgem3_model(cfg):
    from FlagEmbedding import BGEM3FlagModel

    checkpoint = require_config_value(cfg, "baselines.checkpoint")
    use_fp16 = bool(get_config_value(cfg, "baselines.use_fp16", True))
    devices = get_config_value(cfg, "baselines.devices", "cuda:0")
    return BGEM3FlagModel(checkpoint, use_fp16=use_fp16, devices=devices)


def _bgem3_tokenize_with_spans_factory(cfg):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(require_config_value(cfg, "baselines.tokenizer_name"), use_fast=True)
    return _tokenize_with_spans_factory(tokenizer)


def _load_colbertv2_checkpoint(checkpoint_path):
    from colbert.infra import ColBERTConfig
    from colbert.modeling.checkpoint import Checkpoint
    from transformers.modeling_utils import PreTrainedModel

    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}

    config = ColBERTConfig(checkpoint=checkpoint_path)
    return Checkpoint(checkpoint_path, colbert_config=config)


def _colbertv2_tokenize_with_spans_factory(checkpoint):
    return _tokenize_with_spans_factory(checkpoint.doc_tokenizer.tok)


def _tokenize_with_spans_factory(tokenizer):
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


def _encode_bgem3_doc_chunks(model, chunks, output_path, metadata_path, batch_size, max_length):
    import torch
    from safetensors.torch import save_file

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
    _write_chunks(metadata_path, chunks)
    logger.info("Saved %s and %s", output_path, metadata_path)


def _encode_bgem3_query(model, question, output_path, batch_size, max_length):
    import torch
    from safetensors.torch import save_file

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


def _encode_colbertv2_doc_chunks(checkpoint, chunks, output_path, metadata_path, batch_size):
    import torch
    from safetensors.torch import save_file

    texts = [chunk["text"] for chunk in chunks]
    doc_embs = checkpoint.docFromText(texts, bsize=batch_size, keep_dims="flatten", to_cpu=False)
    if not isinstance(doc_embs, tuple) or len(doc_embs) < 2:
        raise ValueError("Unexpected ColBERTv2 docFromText return value.")
    embs, doclens = doc_embs[:2]
    embs = embs.detach().to(dtype=torch.float32, device="cpu")
    doclens = torch.as_tensor(doclens, dtype=torch.int32, device="cpu")
    save_file({"embeddings": embs, "doclens": doclens}, output_path)
    _write_chunks(metadata_path, chunks)
    logger.info("Saved %s and %s", output_path, metadata_path)


def _encode_colbertv2_query(checkpoint, question, output_path, batch_size):
    import torch
    from safetensors.torch import save_file

    query_emb = checkpoint.queryFromText([question], bsize=batch_size, to_cpu=False)
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


def _write_chunks(path, chunks):
    with path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def _load_samples(cfg, benchmark_name):
    if benchmark_name == "mmlongbench":
        return load_mmlongbench_samples(require_config_value(cfg, "benchmarks.input_path"))
    if benchmark_name == "longdocurl":
        return load_longdocurl_samples(
            require_config_value(cfg, "benchmarks.qa_file"),
            image_prefix=get_config_value(cfg, "benchmarks.image_prefix"),
        )
    raise ValueError(f"Unsupported benchmark: {benchmark_name}")


def _limit_samples(samples, limit):
    if limit is None:
        return samples
    return samples[: int(limit)]
