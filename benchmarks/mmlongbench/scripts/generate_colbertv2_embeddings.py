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

from baselines.utils.benchmarks_related import build_token_chunks_from_pages, load_mmlongbench_ocr_pages


DEFAULT_INPUT_PATH = CODE_DIR / "benchmarks" / "mmlongbench" / "data" / "samples.json"
DEFAULT_DOC_OUTPUT_DIR = CODE_DIR / "benchmarks" / "mmlongbench" / "tmp" / "colbertv2" / "doc_embeddings"
DEFAULT_QUERY_OUTPUT_DIR = CODE_DIR / "benchmarks" / "mmlongbench" / "tmp" / "colbertv2" / "query_embeddings"
DEFAULT_METADATA_OUTPUT_DIR = CODE_DIR / "benchmarks" / "mmlongbench" / "tmp" / "colbertv2" / "chunk_metadata"
DEFAULT_CHECKPOINT = "/root/autodl-tmp/ylz/models/colbertv2.0"

logger = logging.getLogger("generate_mmlongbench_colbertv2_embeddings")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ColBERTv2 chunk and query embeddings for MMLongBench.")
    parser.add_argument("--mode", choices=["doc", "query", "both"], default="both")
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--doc-output-dir", default=str(DEFAULT_DOC_OUTPUT_DIR))
    parser.add_argument("--query-output-dir", default=str(DEFAULT_QUERY_OUTPUT_DIR))
    parser.add_argument("--metadata-output-dir", default=str(DEFAULT_METADATA_OUTPUT_DIR))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--doc-id", action="append", default=None)
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
    parser.add_argument("--max-pages", type=int, default=120)
    parser.add_argument("--ocr-json-dir", default=str(CODE_DIR / "benchmarks" / "mmlongbench" / "tmp" / "pdf_jsons"))
    parser.add_argument("--tmp-dir", default=str(CODE_DIR / "benchmarks" / "mmlongbench" / "tmp"))
    return parser.parse_args()


def resolve_device(device):
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_samples(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def grouped_doc_samples(samples, explicit_doc_ids):
    if explicit_doc_ids:
        allowed = set(explicit_doc_ids)
        return sorted(allowed)
    return sorted({sample["doc_id"] for sample in samples})


def grouped_question_samples(samples, explicit_doc_ids, explicit_question_ids):
    explicit_doc_ids = set(explicit_doc_ids or [])
    explicit_question_ids = set(explicit_question_ids or [])
    by_id = {}
    for sample in samples:
        if explicit_doc_ids and sample["doc_id"] not in explicit_doc_ids:
            continue
        if explicit_question_ids and sample["question_id"] not in explicit_question_ids:
            continue
        by_id.setdefault(sample["question_id"], {
            "question_id": sample["question_id"],
            "doc_id": sample["doc_id"],
            "question": sample["question"],
        })
    return [by_id[key] for key in sorted(by_id)]


def build_cfg(args):
    return {
        "benchmarks": {
            "name": "mmlongbench",
            "tmp_dir": args.tmp_dir,
            "ocr_json_dir": args.ocr_json_dir,
            "max_pages": args.max_pages,
        }
    }


def mmlongbench_doc_key(doc_id):
    filename = Path(str(doc_id)).name
    stem = filename.rsplit(".", 1)[0]
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem).strip("._") or "document"


def load_checkpoint(checkpoint_path, device):
    from colbert.modeling.checkpoint import Checkpoint

    checkpoint = Checkpoint(checkpoint_path)
    checkpoint.colbert.to(device)
    checkpoint.colbert.eval()
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


def encode_doc_chunks(checkpoint, chunks, output_path, metadata_path, batch_size, overwrite):
    if output_path.exists() and metadata_path.exists() and not overwrite:
        logger.info("Skipping existing ColBERTv2 doc embeddings: %s", output_path)
        return "skipped"

    texts = [chunk["text"] for chunk in chunks]
    doc_embs = checkpoint.docFromText(texts, bsize=batch_size, keep_dims=False, to_cpu=True)
    if isinstance(doc_embs, tuple):
        embs, doclens = doc_embs
    else:
        raise ValueError("Unexpected ColBERTv2 docFromText return value.")

    embs = embs.to(dtype=torch.float32, device="cpu")
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

    query_emb = checkpoint.queryFromText([question], bsize=batch_size, to_cpu=True)
    if isinstance(query_emb, torch.Tensor):
        if query_emb.ndim == 3:
            query_emb = query_emb[0]
    else:
        query_emb = torch.as_tensor(query_emb[0] if isinstance(query_emb, (list, tuple)) else query_emb)
    query_emb = query_emb.to(dtype=torch.float32, device="cpu")
    save_file({"query_embedding": query_emb}, output_path)
    logger.info("Saved %s", output_path)
    return "generated"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    device = resolve_device(args.device)
    samples = load_samples(args.input_path)

    doc_ids = grouped_doc_samples(samples, args.doc_id)
    question_samples = grouped_question_samples(samples, args.doc_id, args.question_id)
    if args.limit is not None:
        doc_ids = doc_ids[:args.limit]
        question_samples = question_samples[:args.limit]

    Path(args.doc_output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.query_output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.metadata_output_dir).mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(args.checkpoint, device)
    tokenize_with_spans = tokenize_with_spans_factory(checkpoint)
    cfg = build_cfg(args)
    sample_by_doc = {}
    for sample in samples:
        sample_by_doc.setdefault(sample["doc_id"], sample)

    if args.mode in {"doc", "both"}:
        for doc_id in tqdm(doc_ids, desc="MMLongBench ColBERTv2 doc embeddings"):
            sample = sample_by_doc[doc_id]
            pages, _ = load_mmlongbench_ocr_pages(sample, cfg["benchmarks"])
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
            doc_key = mmlongbench_doc_key(doc_id)
            encode_doc_chunks(
                checkpoint,
                chunks,
                Path(args.doc_output_dir) / f"{doc_key}.safetensors",
                Path(args.metadata_output_dir) / f"{doc_key}.json",
                args.batch_size,
                args.overwrite,
            )

    if args.mode in {"query", "both"}:
        for sample in tqdm(question_samples, desc="MMLongBench ColBERTv2 query embeddings"):
            encode_query(
                checkpoint,
                sample["question"],
                Path(args.query_output_dir) / f"{sample['question_id']}.safetensors",
                args.batch_size,
                args.overwrite,
            )


if __name__ == "__main__":
    main()
