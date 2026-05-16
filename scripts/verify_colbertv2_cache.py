#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
from baselines.utils.benchmarks_related import colbertv2_doc_cache_variant, colbertv2_query_cache_variant

def parse_args():
    parser = argparse.ArgumentParser(description="Verify ColBERTv2 cache coverage before running evaluation.")
    parser.add_argument("--benchmark", choices=["mmlongbench", "longdocurl"], required=True)
    parser.add_argument("--input-path", required=True, help="Sample file to be evaluated.")
    parser.add_argument("--doc-embeddings-root", required=True)
    parser.add_argument("--query-embeddings-root", required=True)
    parser.add_argument("--chunk-metadata-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--chunk-overlap", type=int, required=True)
    parser.add_argument("--allow-cross-page", required=True)
    parser.add_argument("--max-cross-pages", default=None)
    parser.add_argument("--report-path", default=None, help="Optional JSON report path.")
    return parser.parse_args()


def load_samples(benchmark, input_path):
    if benchmark == "mmlongbench":
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)

    samples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            sample = json.loads(line)
            sample.setdefault("question_id", idx)
            samples.append(sample)
    return samples


def required_paths(benchmark, sample):
    if benchmark == "mmlongbench":
        filename = Path(str(sample["doc_id"])).name
        stem = filename.rsplit(".", 1)[0]
        doc_key = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem).strip("._") or "document"
        query_key = str(sample["question_id"])
    else:
        doc_key = str(sample["question_id"])
        query_key = str(sample["question_id"])
    return doc_key, query_key


def verify(args):
    samples = load_samples(args.benchmark, args.input_path)
    doc_root = Path(args.doc_embeddings_root) / colbertv2_doc_cache_variant(
        args.checkpoint,
        args.chunk_size,
        args.chunk_overlap,
        str(args.allow_cross_page).lower() == "true",
        None if args.max_cross_pages in (None, "", "null", "None") else int(args.max_cross_pages),
    )
    query_root = Path(args.query_embeddings_root) / colbertv2_query_cache_variant(args.checkpoint)
    meta_root = Path(args.chunk_metadata_root) / colbertv2_doc_cache_variant(
        args.checkpoint,
        args.chunk_size,
        args.chunk_overlap,
        str(args.allow_cross_page).lower() == "true",
        None if args.max_cross_pages in (None, "", "null", "None") else int(args.max_cross_pages),
    )

    missing = []
    for sample in samples:
        doc_key, query_key = required_paths(args.benchmark, sample)
        doc_path = doc_root / f"{doc_key}.safetensors"
        query_path = query_root / f"{query_key}.safetensors"
        meta_path = meta_root / f"{doc_key}.json"
        missing_fields = []
        if not doc_path.exists():
            missing_fields.append("doc_embeddings")
        if not query_path.exists():
            missing_fields.append("query_embeddings")
        if not meta_path.exists():
            missing_fields.append("chunk_metadata")
        if missing_fields:
            missing.append({
                "sample_key": sample.get("question_id", sample.get("doc_id")),
                "doc_key": doc_key,
                "query_key": query_key,
                "missing": missing_fields,
            })

    report = {
        "benchmark": args.benchmark,
        "input_path": args.input_path,
        "sample_count": len(samples),
        "missing_count": len(missing),
        "missing": missing,
    }
    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"benchmark={args.benchmark}")
    print(f"sample_count={len(samples)}")
    print(f"missing_count={len(missing)}")
    if missing:
        print("first_missing_examples:")
        for item in missing[:10]:
            print(json.dumps(item, ensure_ascii=False))
        return 1
    print("all_required_colbertv2_cache_files_present")
    return 0


if __name__ == "__main__":
    raise SystemExit(verify(parse_args()))
