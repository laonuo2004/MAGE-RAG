# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python benchmark harness for long-document multimodal QA. The main entrypoint is `main.py`, which loads Hydra configs from `configs/`. Baselines live in `baselines/`, benchmark routing in `benchmarks/`, and shared helpers in `utils/`. Root tests are in `tests/`. Benchmark-specific code and assets are under `benchmarks/longdocurl/` and `benchmarks/mmlongbench/`; generated outputs belong under each benchmark's `evaluation_results/` or Hydra `outputs/`. The vendored M3DocRAG implementation is in `baselines/m3docrag/`.

## Build, Test, and Development Commands

- Use `conda logma-rag-py12` environment for development, which includes all dependencies.
- `python main.py`: run the default Hydra configuration from `configs/config.yaml`.
- `python main.py benchmarks=mmlongbench baselines=image`: run a specific benchmark and baseline override.
- `python main.py --multirun benchmarks=longdocurl,mmlongbench baselines=bm25`: launch a Hydra sweep.
- `bash scripts/run_bm25_sweep.sh`: run the repository's BM25 sweep presets.
- `pytest tests`: run the root unit tests for config handling, context builders, and benchmark result validation.
- `bash scripts/serve_litellm.sh configs/litellm_config.yaml`: start the local LiteLLM OpenAI-compatible proxy for API-backed evaluation.

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation, explicit imports, and small functions that follow existing module boundaries. Prefer `snake_case` for functions, variables, config keys, and tests; use `PascalCase` for classes. Keep Hydra config groups lowercase and descriptive, for example `configs/baselines/bm25.yaml`. Shell scripts should use `#!/usr/bin/env bash` plus `set -euo pipefail`, matching existing scripts.

我们统一使用绝对路径导入而不使用相对路径导入。

## Testing Guidelines

Tests use `pytest` while many files are written with `unittest.TestCase`. Add or update `tests/test_*.py` for changes to routing, context construction, config validation, or result parsing. Keep fixtures lightweight with temporary directories and tiny synthetic images or JSON records. Run `pytest tests` before opening a PR; run targeted tests such as `pytest tests/test_context_builders.py` while iterating.

## Commit & Pull Request Guidelines

The current git history uses short, direct commit messages, including brief Chinese summaries such as `bm参数`; keep messages concise and focused on one change. PRs should describe the benchmark or baseline affected, list the commands run, and mention any required local services such as vLLM or LiteLLM. Include sample output paths or metric files for evaluation changes, but avoid committing large generated results unless they are intentional reference artifacts.

## Security & Configuration Tips

Secrets should stay in `.env`, which `main.py` loads from the repository root. Do not hard-code API keys, local model tokens, or private server paths in configs. Prefer Hydra overrides for experiment-specific values, and keep machine-specific service settings in local scripts or ignored environment files.
