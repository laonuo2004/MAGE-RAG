#!/usr/bin/env python3

"""Remove JSONL rows whose `pred` field is `Fail to extract`."""

import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


FAILED_PRED = "Fail to extract"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete JSONL rows whose pred field equals 'Fail to extract'."
    )
    parser.add_argument("--results_file", help="Path to the input JSONL results file.")
    parser.add_argument(
        "--output_file",
        default="",
        help="Optional output path. If omitted, the input file is overwritten in place.",
    )
    return parser.parse_args()


def should_keep(line: str) -> bool:
    if not line.strip():
        return False

    sample = json.loads(line)
    return sample.get("pred") != FAILED_PRED


def clean_jsonl(input_path: Path, output_path: Path) -> tuple[int, int]:
    kept = 0
    removed = 0

    with input_path.open("r", encoding="utf-8") as reader, output_path.open(
        "w", encoding="utf-8"
    ) as writer:
        for line_number, line in enumerate(reader, start=1):
            try:
                keep = should_keep(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {input_path}: {exc.msg}"
                ) from exc

            if keep:
                writer.write(line if line.endswith("\n") else line + "\n")
                kept += 1
            else:
                removed += 1

    return kept, removed


def main() -> None:
    args = parse_args()
    input_path = Path(args.results_file).expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    output_path = Path(args.output_file).expanduser().resolve() if args.output_file else input_path

    if output_path == input_path:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=str(input_path.parent),
            prefix=f".{input_path.name}.",
            suffix=".tmp",
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            kept, removed = clean_jsonl(input_path, tmp_path)
            os.replace(tmp_path, input_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        kept, removed = clean_jsonl(input_path, output_path)

    print(f"Kept={kept} Removed={removed} Output={output_path}")


if __name__ == "__main__":
    main()
