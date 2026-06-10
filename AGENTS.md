# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python benchmark harness for long-document multimodal QA. The main entrypoint is `main.py`, which loads Hydra configs from `configs/`. Baselines live in `baselines/`, benchmark routing in `benchmarks/`, benchmark data/cache helpers in `benchmarks/utils/`, and shared helpers in `utils/`. Root tests are in `tests/`. Benchmark-specific code and assets are under `benchmarks/longdocurl/` and `benchmarks/mmlongbench/`; evidence graph utilities live under `benchmarks/evidence_graph/`. Result analysis code lives in `analysis/`; generated analysis cache belongs in `analysis_cache/`. Generated evaluation artifacts belong under each benchmark's `evaluation_results/`, repository `results/`, or Hydra `outputs/`. The vendored M3DocRAG implementation is in `baselines/m3docrag/`.

## Build, Test, and Development Commands

- Use the `logma-rag-py12` Conda environment for development. Its Python path is `/root/autodl-tmp/conda/envs/logma-rag-py12/bin/python`; prefer this absolute path in non-interactive or elevated commands.
- `/root/autodl-tmp/conda/envs/logma-rag-py12/bin/python main.py`: run the default Hydra configuration from `configs/config.yaml`.
- `/root/autodl-tmp/conda/envs/logma-rag-py12/bin/python main.py benchmarks=mmlongbench baselines=image`: run a specific benchmark and baseline override.
- `/root/autodl-tmp/conda/envs/logma-rag-py12/bin/python main.py --multirun benchmarks=longdocurl,mmlongbench baselines=bm25`: launch a Hydra sweep.
- `bash scripts/run_bm25_sweep.sh`: run the repository's BM25 sweep presets.
- `bash scripts/run_build_longdocurl.sh` / `bash scripts/run_build_mmlongbench.sh`: build benchmark-specific preprocessing artifacts.
- `bash scripts/run_bgem3_longdocurl_sweep.sh` / `bash scripts/run_bgem3_mmlongbench_sweep.sh`: run BGE-M3 sweep presets.
- `bash scripts/run_m3docrag.sh` or `bash scripts/run_m3docrag-iterate.sh`: run M3DocRAG baselines.
- `bash scripts/serve_results_dashboard.sh`: serve the local results dashboard.
- `/root/autodl-tmp/conda/envs/logma-rag-py12/bin/python -m pytest tests`: run the root unit tests when stabilizing a complete change set.
- `/root/autodl-tmp/conda/envs/logma-rag-py12/bin/python -m pytest tests/test_data_utils.py tests/test_benchmark_adapters.py`: run focused tests for benchmark path helpers and adapter behavior.
- `/root/autodl-tmp/conda/envs/logma-rag-py12/bin/python -m pytest tests/test_results_analysis.py`: run focused tests for analysis output handling.
- `bash scripts/serve_litellm.sh configs/litellm_config.yaml`: start the local LiteLLM OpenAI-compatible proxy for API-backed evaluation.
- `bash scripts/serve_colpali_vllm.sh`, `bash scripts/serve_qwen25_vl_vllm.sh`, or `bash scripts/serve_qwen3_vl_vllm.sh`: start local vLLM services used by retrieval or API-backed evaluation.

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation and explicit imports. Prefer `snake_case` for functions, variables, config keys, and tests; use `PascalCase` for classes. Keep functions single-purpose; when a function grows beyond roughly 40 lines or mixes I/O, parsing, scoring, and service calls, split it unless the surrounding module clearly uses a different local pattern. Keep Hydra config groups lowercase and descriptive, for example `configs/baselines/bm25.yaml`. Shell scripts should use `#!/usr/bin/env bash` plus `set -euo pipefail`, matching existing scripts.

我们统一使用绝对路径导入而不使用相对路径导入。

Benchmark path conventions should live in code constants and helpers rather than ad hoc Hydra branches; prefer `benchmarks/utils/data_utils.py` for shared benchmark data and cache path logic. Keep benchmark-specific scoring and result parsing inside adapter-owned code paths.

## Testing Guidelines

Tests use `pytest` while many files are written with `unittest.TestCase`. Add or update `tests/test_*.py` for changes to routing, context construction, config validation, or result parsing. Keep fixtures lightweight with temporary directories and tiny synthetic images or JSON records. During research-first migrations, run the targeted tests that cover the new standard path you changed, such as `/root/autodl-tmp/conda/envs/logma-rag-py12/bin/python -m pytest tests/test_context_builders.py`. Run `/root/autodl-tmp/conda/envs/logma-rag-py12/bin/python -m pytest tests` when stabilizing a complete change set; if failures come from legacy paths that have not been migrated yet, document the failing command and failure instead of adding compatibility layers for the old API.

For benchmark changes, trace the active runtime path from `main.py` through `benchmarks.wrapper`, `benchmarks.runner`, `benchmarks.adapters`, and the selected baseline context builder before changing shared interfaces. For service debugging, do not rely on PM2 state alone; compare process state and listening ports, then validate with model-scoped health checks or a small meaningful traffic burst.

## Commit & Pull Request Guidelines

The current git history uses short, direct commit messages, including brief Chinese summaries such as `bm参数`; keep messages concise and focused on one change. PRs should describe the benchmark or baseline affected, list the commands run, and mention any required local services such as vLLM or LiteLLM. Include sample output paths or metric files for evaluation changes, but avoid committing large generated results unless they are intentional reference artifacts.

## Security & Configuration Tips

Secrets should stay in `.env`, which `main.py` loads from the repository root. Do not hard-code API keys, local model tokens, or private server paths in configs. Prefer Hydra overrides for experiment-specific values, and keep machine-specific service settings in local scripts or ignored environment files.

## 注意事项

- 当前服务器沙箱服务可能存在异常；如果 Codex sandbox 或 bwrap 导致命令失败，使用平台提供的 escalation/approval 机制重跑命令，不要手动改写为 `sudo`。需要运行 Python 或 pytest 时，显式使用 `/root/autodl-tmp/conda/envs/logma-rag-py12/bin/python`，避免绕过 Conda 环境。
- 在进行方案设计时，以**结果**为优先导向，而不优先考虑成本等工程因素，以检验方案的最佳潜力。
- 在进行诸如代码修改、文档更新、论文撰写等相关工作时，**不要考虑任何向后兼容性问题**！我们只允许最新的标准和最佳实践，过时的内容不允许存在于仓库中：
  - 对于代码，不要为旧接口、旧配置或旧数据格式编写兼容层、fallback、迁移提示异常或双路径逻辑；只保证本次替换后的新标准路径一致可运行。如果未迁移的旧路径在测试或运行中自然失败，记录触发命令和失败现象，后续按新标准逐一更新相关代码。
  - 对于文档与论文，如果存在过时的表述，在更新时不应该出现类似于 "Stage II 默认运行在线 agent，不再保留 `online_agent=false` 的 retrieval-only 分支，也不再保留 `graph_escape` 等旧配置" 这种暗示该文档、论文**曾经做过更改**的描述，而应该直接以最新的设计方案进行描述，完全不提及任何过时的设计或配置。这样做是为了让文档和论文保持清晰、简洁，并且避免引入任何可能引起混淆的历史信息。
