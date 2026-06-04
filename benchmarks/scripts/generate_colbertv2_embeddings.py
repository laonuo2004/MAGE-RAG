#!/usr/bin/env python3
import argparse
import runpy
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--benchmark", choices=["longdocurl", "mmlongbench"], required=True)
    args, remaining = parser.parse_known_args()
    target = Path(__file__).resolve().parents[1] / args.benchmark / "scripts" / "generate_colbertv2_embeddings.py"
    sys.argv = [str(target), *remaining]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
