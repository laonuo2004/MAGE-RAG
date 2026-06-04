import argparse
from pathlib import Path

from omegaconf import OmegaConf

from benchmarks.utils.embedding_cache import generate_embedding_cache


def run_embedding_cache_cli(*, baseline_name, benchmark_name, defaults):
    args = _parse_args(defaults)
    cfg = _build_cfg(args, baseline_name=baseline_name, benchmark_name=benchmark_name)
    generate_embedding_cache(
        cfg,
        baseline_name=baseline_name,
        mode=args.mode,
        overwrite=args.overwrite,
        doc_ids=args.doc_id or args.doc_no,
        question_ids=args.question_id,
    )


def _parse_args(defaults):
    parser = argparse.ArgumentParser(description=defaults["description"])
    parser.add_argument("--mode", choices=["doc", "query", "both"], default="both")
    parser.add_argument("--input-path", default=str(defaults["input_path"]))
    parser.add_argument("--doc-output-dir", default=str(defaults["doc_output_dir"]))
    parser.add_argument("--query-output-dir", default=str(defaults["query_output_dir"]))
    parser.add_argument("--metadata-output-dir", default=str(defaults["metadata_output_dir"]))
    parser.add_argument("--checkpoint", default=str(defaults["checkpoint"]))
    parser.add_argument("--tokenizer-name", default=str(defaults.get("tokenizer_name", defaults["checkpoint"])))
    parser.add_argument("--devices", default=defaults.get("devices", "cuda:0"))
    parser.add_argument("--device", default=defaults.get("device", "auto"))
    parser.add_argument("--doc-id", action="append", default=None)
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
    parser.add_argument("--max-pages", type=int, default=defaults.get("max_pages", 120))
    parser.add_argument("--tmp-dir", default=str(defaults.get("tmp_dir", "")))
    parser.add_argument("--ocr-json-dir", default=str(defaults.get("ocr_json_dir", "")))
    parser.add_argument("--image-prefix", default=str(defaults.get("image_prefix", "")))
    parser.add_argument("--mineru-dir", default=str(defaults.get("mineru_dir", "")))
    parser.add_argument("--mode-name", default="dense")
    parser.add_argument("--text-source", default="ocr")
    parser.add_argument("--use-fp16", action="store_true", default=True)
    parser.add_argument("--no-use-fp16", action="store_false", dest="use_fp16")
    return parser.parse_args()


def _build_cfg(args, *, baseline_name, benchmark_name):
    baseline_cfg = {
        "name": baseline_name,
        "checkpoint": args.checkpoint,
        "tokenizer_name": args.tokenizer_name,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "use_fp16": args.use_fp16,
        "devices": args.devices,
        "device": args.device,
        "params": {
            "mode": args.mode_name,
            "text_source": args.text_source,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "allow_cross_page": args.allow_cross_page,
            "max_cross_pages": args.max_cross_pages,
        },
    }
    benchmark_cfg = {
        "name": benchmark_name,
        "input_path": args.input_path,
        "qa_file": args.input_path,
        "tmp_dir": args.tmp_dir,
        "ocr_json_dir": _none_if_empty(args.ocr_json_dir),
        "image_prefix": _none_if_empty(args.image_prefix),
        "mineru_dir": _none_if_empty(args.mineru_dir),
        "max_pages": args.max_pages,
    }
    cfg = {
        "baselines": baseline_cfg,
        "benchmarks": benchmark_cfg,
        "cache_output_dirs": {
            "doc_embeddings": str(Path(args.doc_output_dir)),
            "query_embeddings": str(Path(args.query_output_dir)),
            "chunk_metadata": str(Path(args.metadata_output_dir)),
        },
    }
    if args.limit is not None:
        cfg["cache_sample_limit"] = args.limit
    return OmegaConf.create(cfg)


def _none_if_empty(value):
    return None if value in ("", None) else value
