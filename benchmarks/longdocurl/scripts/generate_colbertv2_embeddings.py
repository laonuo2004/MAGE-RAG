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

from baselines.utils.benchmarks_related import build_token_chunks_from_pages, load_longdocurl_ocr_pages


DEFAULT_INPUT_PATH = CODE_DIR / "benchmarks" / "longdocurl" / "data" / "LongDocURL.jsonl"
DEFAULT_DOC_OUTPUT_DIR = CODE_DIR / "benchmarks" / "longdocurl" / "tmp" / "colbertv2" / "doc_embeddings"
DEFAULT_QUERY_OUTPUT_DIR = CODE_DIR / "benchmarks" / "longdocurl" / "tmp" / "colbertv2" / "query_embeddings"
DEFAULT_METADATA_OUTPUT_DIR = CODE_DIR / "benchmarks" / "longdocurl" / "tmp" / "colbertv2" / "chunk_metadata"
DEFAULT_CHECKPOINT = "/root/autodl-tmp/ylz/models/colbertv2.0"
DEFAULT_OCR_JSON_DIR = CODE_DIR / "benchmarks" / "longdocurl" / "data" / "pdf_jsons" / "4000-4999"
DEFAULT_IMAGE_PREFIX = CODE_DIR / "benchmarks" / "longdocurl" / "data" / "pdf_pngs" / "4000-4999"

logger = logging.getLogger("generate_longdocurl_colbertv2_embeddings")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ColBERTv2 chunk and query embeddings for LongDocURL.")
    parser.add_argument("--mode", choices=["doc", "query", "both"], default="both")
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--doc-output-dir", default=str(DEFAULT_DOC_OUTPUT_DIR))
    parser.add_argument("--query-output-dir", default=str(DEFAULT_QUERY_OUTPUT_DIR))
    parser.add_argument("--metadata-output-dir", default=str(DEFAULT_METADATA_OUTPUT_DIR))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--doc-no", action="append", default=None)
    parser.add_argument("--question-id", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--chunk-overlap", type=int, default=20)
    parser.add_argument("--allow-cross-page", action="store_true", default=True)
    parser.add_argument("--no-allow-cross-page", action="store_false", dest="allow_cross_page")
    parser.add_argument("--max-cross-pages", type=int, default=None)
    parser.add_argument("--ocr-json-dir", default=str(DEFAULT_OCR_JSON_DIR))
    parser.add_argument("--image-prefix", default=str(DEFAULT_IMAGE_PREFIX))
    return parser.parse_args()


def resolve_device(device):
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


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


def load_checkpoint(checkpoint_path, device):
    from colbert.infra import ColBERTConfig
    from transformers.modeling_utils import PreTrainedModel
    from colbert.modeling.checkpoint import Checkpoint

    # ColBERT's dynamically created HuggingFace wrapper lags behind newer
    # Transformers, which now expects all_tied_weights_keys during model
    # finalization. Patching the base class keeps the dynamic subclass happy.
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}

    config = ColBERTConfig(checkpoint=checkpoint_path)
    checkpoint = Checkpoint(checkpoint_path, colbert_config=config)
    return checkpoint


def tokenize_with_spans_factory(checkpoint):
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


def build_cfg(args):
    return {
        "benchmarks": {
            "name": "longdocurl",
            "ocr_json_dir": args.ocr_json_dir,
            "image_prefix": args.image_prefix,
        }
    }


def encode_doc_chunks(checkpoint, chunks, output_path, metadata_path, batch_size, overwrite):
    if output_path.exists() and metadata_path.exists() and not overwrite:
        logger.info("Skipping existing ColBERTv2 doc embeddings: %s", output_path)
        return "skipped"

    texts = [chunk["text"] for chunk in chunks]
    doc_embs = checkpoint.docFromText(texts, bsize=batch_size, keep_dims="flatten", to_cpu=False)
    if not isinstance(doc_embs, tuple) or len(doc_embs) < 2:
        raise ValueError("Unexpected ColBERTv2 docFromText return value.")
    embs, doclens = doc_embs[:2]

    embs = embs.detach().to(dtype=torch.float32, device="cpu")
    doclens = torch.as_tensor(doclens, dtype=torch.int32, device="cpu")
    save_file({"embeddings": embs, "doclens": doclens}, output_path)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info("Saved %s and %s", output_path, metadata_path)
    return "generated"


def encode_query(checkpoint, question, output_path, batch_size, overwrite):
    if output_path.exists() and not overwrite:
        logger.info("Skipping existing ColBERTv2 query embeddings: %s", output_path)
        return "skipped"

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
    return "generated"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    device = resolve_device(args.device)
    samples = load_samples(args.input_path, args.image_prefix)
    selected = select_samples(samples, args.doc_no, args.question_id, args.limit)

    Path(args.doc_output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.query_output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.metadata_output_dir).mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(args.checkpoint, device)
    tokenize_with_spans = tokenize_with_spans_factory(checkpoint)
    cfg = build_cfg(args)

    for sample in tqdm(selected, desc="LongDocURL ColBERTv2 embeddings"):
        if args.mode in {"doc", "both"}:
            pages, _ = load_longdocurl_ocr_pages(sample, cfg["benchmarks"])
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
                checkpoint,
                chunks,
                Path(args.doc_output_dir) / f"{sample['question_id']}.safetensors",
                Path(args.metadata_output_dir) / f"{sample['question_id']}.json",
                args.batch_size,
                args.overwrite,
            )

        if args.mode in {"query", "both"}:
            encode_query(
                checkpoint,
                sample["question"],
                Path(args.query_output_dir) / f"{sample['question_id']}.safetensors",
                args.batch_size,
                args.overwrite,
            )


if __name__ == "__main__":
    main()
