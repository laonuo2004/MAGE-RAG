#!/usr/bin/env python3
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal BGEM3 smoke test.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--use-fp16", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()

    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel(args.checkpoint, use_fp16=args.use_fp16)
    outputs = model.encode(
        ["hello world", "retrieval test"],
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )

    summary = {
        "output_type": type(outputs).__name__,
        "keys": list(outputs.keys()) if isinstance(outputs, dict) else None,
        "dense_type": type(outputs["dense_vecs"]).__name__ if isinstance(outputs, dict) and "dense_vecs" in outputs else None,
        "dense_count": len(outputs["dense_vecs"]) if isinstance(outputs, dict) and "dense_vecs" in outputs else None,
        "dense_dim": len(outputs["dense_vecs"][0]) if isinstance(outputs, dict) and "dense_vecs" in outputs and len(outputs["dense_vecs"]) else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
