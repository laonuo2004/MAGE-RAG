#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_PATH = CODE_DIR / "benchmarks" / "mmlongbench" / "data" / "samples.json"
DEFAULT_PREFIX = "q"
DEFAULT_PADDING = 6


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add a unique question_id field to every MMLongBench sample."
    )
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH), help="Path to samples.json.")
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional output path. Defaults to overwriting the input file.",
    )
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="Prefix for generated question_id values.")
    parser.add_argument("--padding", type=int, default=DEFAULT_PADDING, help="Zero padding width for generated ids.")
    return parser.parse_args()


def load_samples(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if not isinstance(samples, list):
        raise ValueError(f"Expected a JSON array at {input_path}, got {type(samples).__name__}.")
    return samples


def allocate_question_ids(samples, prefix, padding):
    used_ids = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"Sample at index {index} is not an object: {type(sample).__name__}.")
        existing_id = sample.get("question_id")
        if existing_id is None or existing_id == "":
            continue
        if existing_id in used_ids:
            raise ValueError(f"Duplicate question_id found in input: {existing_id!r}")
        used_ids.add(existing_id)

    next_number = 1
    for sample in samples:
        if sample.get("question_id") not in (None, ""):
            continue

        while True:
            candidate = f"{prefix}{next_number:0{padding}d}"
            next_number += 1
            if candidate not in used_ids:
                sample["question_id"] = candidate
                used_ids.add(candidate)
                break


def write_samples(output_path, samples):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path) if args.output_path else input_path

    samples = load_samples(input_path)
    allocate_question_ids(samples, args.prefix, args.padding)
    write_samples(output_path, samples)


if __name__ == "__main__":
    main()