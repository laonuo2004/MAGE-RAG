# MMLongBench Benchmark Usage Guide

## 1. Scope

This document describes how the current `mmlongbench` benchmark pipeline is used in this repo, what has been modified relative to the upstream benchmark, and how to integrate later baseline methods into the same evaluation workflow.

This guide is written for the current code under:

- `code/benchmarks/mmlongbench/run_api.py`
- `code/benchmarks/mmlongbench/run_api_text.py`
- `code/benchmarks/mmlongbench/eval/extract_answer.py`
- `code/benchmarks/mmlongbench/eval/eval_score.py`
- `code/benchmarks/mmlongbench/env_utils.py`
- `code/benchmarks/mmlongbench/route_utils.py`

The goal of the current implementation is:

1. Run the benchmark with OpenAI-compatible APIs such as vLLM / LiteLLM / OpenRouter.
2. Support both image-input and OCR/text-input routes.
3. Support resumable execution.
4. Support retrying failed samples without manually deleting output files.
5. Support multi-route inference with context-aware fallback.
6. Support limited concurrency for faster benchmarking.

---

## 2. Directory Layout

Current relevant files:

```text
code/benchmarks/mmlongbench/
├── data/
│   ├── samples.json
│   └── documents/               # PDF files
├── eval/
│   ├── eval_score.py
│   ├── extract_answer.py
│   └── prompt_for_answer_extraction.md
├── env_utils.py
├── route_utils.py
├── run_api.py                   # image route
├── run_api_text.py              # OCR/text route
├── .env.mmlongbench.example
└── MMLONGBENCH_BENCHMARK_GUIDE.md
```

Runtime-generated directories:

```text
tmp/                            # rendered PDF page images
results/                        # JSON outputs + txt summaries
logs/                           # tee logs
```

---

## 3. Two Benchmark Routes

### 3.1 Image Route

Entry:

- `run_api.py`

Pipeline:

1. Load benchmark samples from `data/samples.json`.
2. For each sample, open the corresponding PDF from `data/documents/`.
3. Render the first `max_pages` pages into images using PyMuPDF.
4. Send the question plus all selected page images to a multimodal model.
5. Run answer extraction on the long response.
6. Score against benchmark gold answers.
7. Persist progress to JSON after each completed sample.

This route is the main route for `mmlongbench`, because many questions rely on charts, tables, and layout structure.

### 3.2 OCR/Text Route

Entry:

- `run_api_text.py`

Pipeline:

1. Load benchmark samples from `data/samples.json`.
2. Open PDF pages with PyMuPDF.
3. Extract page text via `page.get_text("text")`.
4. Concatenate page text into a long prompt.
5. Ask a language model to answer from the OCR/text prompt.
6. Run answer extraction on the long response.
7. Score against gold answers.

This route is mainly a baseline / control route. It is expected to underperform on strongly visual questions.

---

## 4. Key Parameters

### 4.1 Shared Parameters

- `--input_path`
  Path to benchmark samples JSON.

- `--document_path`
  Directory containing benchmark PDFs.

- `--model_name`
  Main generation model name.

- `--base_url`
  Base URL for a single OpenAI-compatible backend.

- `--api_key`
  API key for the backend.

- `--extractor_model_name`
  Model used for the second-stage answer extraction.

- `--extractor_base_url`
  Base URL used for the extractor.

- `--extractor_api_key`
  API key used for the extractor.

- `--max_try`
  Maximum number of retries before a request is considered failed.

- `--num_workers`
  Concurrency for sample-level execution.

- `--output_path`
  Result JSON file path.

- `--limit`
  Number of samples to run for smoke testing.

### 4.2 Image-Route-Specific

- `--max_pages`
  Current logic selects the first `N` pages of the PDF.

- `--resolution`
  PDF rendering DPI.

Important note:

- `resolution=144` has roughly 4x the pixel count of `resolution=72`.
- Higher resolution improves readability but greatly increases visual token load.

### 4.3 OCR/Text-Route-Specific

- `--max_pages`
  Number of leading pages whose text is concatenated into the prompt.

---

## 5. Current Resume Logic

The original benchmark had only partial save-and-skip logic. This repo now uses merged resume logic:

1. Load the target benchmark set from `samples.json`.
2. Apply `--limit` to determine the target slice.
3. If an existing result file is present, merge previous outputs by sample key:
   - `doc_id`
   - `question`
   - `answer`
   - `answer_format`
4. Skip only samples that are truly completed.
5. Retry samples marked as failed.

This means:

- increasing `--limit` later works correctly
- failed records do not require manual deletion
- rerunning the same output file continues unfinished work

Failure-related fields:

- `status`
  - `completed`
  - `failed_generation`
  - `failed_extraction`

- `failure_stage`
  - `generation`
  - `extraction`

- `error`
  Stores the request error when available

---

## 6. Multi-Route Fallback Logic

### 6.1 Motivation

LiteLLM unified routing on `:4000` is useful operationally, but it does not always automatically route long requests to larger-context backends in the desired way.

For stable benchmarking, this repo adds benchmark-side route fallback.

### 6.2 Route Configuration

Configured through environment variables or CLI:

- `ROUTE_BASE_URLS`
- `ROUTE_MODEL_NAMES`
- `ROUTE_API_KEYS`
- `ROUTE_LABELS`
- `ROUTE_MAX_MODEL_LENS`

Typical ordering:

1. throughput route 1
2. throughput route 2
3. longctx route 1
4. longctx route 2
5. maxctx route

Example:

```bash
ROUTE_BASE_URLS=http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1,http://127.0.0.1:8004/v1,http://127.0.0.1:8000/v1
ROUTE_MODEL_NAMES=/data2/jiangjiaqi/ylz/Qwen2.5-VL-7B-Instruct,/data2/jiangjiaqi/ylz/Qwen2.5-VL-7B-Instruct,/data2/jiangjiaqi/ylz/Qwen2.5-VL-7B-Instruct,/data2/jiangjiaqi/ylz/Qwen2.5-VL-7B-Instruct,/data2/jiangjiaqi/ylz/Qwen2.5-VL-7B-Instruct
ROUTE_API_KEYS=sk-123,sk-123,sk-123,sk-123,sk-123
ROUTE_LABELS=tp1,tp2,long1,long2,max1
ROUTE_MAX_MODEL_LENS=32768,32768,65536,65536,128000
```

### 6.3 Fallback Trigger

The route fallback logic watches for context-overflow-style errors such as:

- `Input length ... exceeds model's maximum context length`
- `maximum context length`
- `too many tokens`

If such an error appears on a smaller route, the benchmark attempts the next larger route.

This is implemented in `route_utils.py` and consumed in:

- `run_api.py`
- `run_api_text.py`

### 6.4 Route Metadata in Output

Each sample stores:

- `used_route_label`
- `used_base_url`
- `used_model_name`
- `used_route_max_model_len`

This is useful for later analysis and group meetings.

---

## 7. Concurrency Logic

Both routes support sample-level parallelism via:

- `--num_workers`

Execution model:

1. Worker threads process samples concurrently.
2. The main thread collects finished futures.
3. After each completed sample, the main thread updates the result array and writes the JSON file.

Advantages:

1. Better throughput than pure serial execution.
2. Safer than concurrent file writes.
3. Resume remains stable.

Suggested values:

- smoke: `--num_workers 1` or `2`
- moderate run: `--num_workers 2` or `4`
- only increase further after service stability is verified

---

## 8. Timing Logs

Both routes now print timing breakdowns:

- `prepare`
- `generation`
- `extraction`
- `total`

Interpretation:

- `prepare`
  - image route: PDF render, caching, base64 preparation
  - text route: OCR/text extraction and prompt assembly

- `generation`
  - main model response time

- `extraction`
  - second-stage extractor response time

- `total`
  - total sample processing time

This helps diagnose whether runtime is dominated by:

1. PDF preprocessing
2. model inference
3. extractor stage

---

## 9. Why Image Tokens Become Very Large

The text portion of the prompt is usually just the question itself.

Large input lengths mainly come from:

1. number of PDF pages converted to images
2. image resolution
3. page complexity (dense tables / charts / small text)

Important:

- image route sends a single message with the question plus multiple page images
- visual token load dominates text token load

Current page-selection logic is simple:

- always use the first `max_pages` pages

This is enough to run the benchmark pipeline, but not yet an optimal retrieval strategy.

---

## 10. Why OCR/Text Often Returns `Not answerable`

This is expected more often in `mmlongbench` than in some other benchmarks because:

1. many questions depend on tables, charts, or layout
2. OCR/text route removes visual structure
3. the prompt explicitly allows `Not answerable` when evidence is insufficient

Therefore:

- image route is the primary route
- OCR/text route is mainly a baseline/control route

---

## 11. Example `.env.mmlongbench` Configs

### 11.1 Direct Multi-Route Version (Recommended Current Setup)

```bash
MODEL_NAME=/data2/jiangjiaqi/ylz/Qwen2.5-VL-7B-Instruct
OPENROUTER_API_KEY=sk-123

ROUTE_BASE_URLS=http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1,http://127.0.0.1:8004/v1,http://127.0.0.1:8000/v1
ROUTE_MODEL_NAMES=/data2/jiangjiaqi/ylz/Qwen2.5-VL-7B-Instruct,/data2/jiangjiaqi/ylz/Qwen2.5-VL-7B-Instruct,/data2/jiangjiaqi/ylz/Qwen2.5-VL-7B-Instruct,/data2/jiangjiaqi/ylz/Qwen2.5-VL-7B-Instruct,/data2/jiangjiaqi/ylz/Qwen2.5-VL-7B-Instruct
ROUTE_API_KEYS=sk-123,sk-123,sk-123,sk-123,sk-123
ROUTE_LABELS=tp1,tp2,long1,long2,max1
ROUTE_MAX_MODEL_LENS=32768,32768,65536,65536,128000

EXTRACTOR_MODEL_NAME=/data2/jiangjiaqi/ylz/Qwen2.5-VL-7B-Instruct
EXTRACTOR_BASE_URL=http://127.0.0.1:8002/v1
EXTRACTOR_API_KEY=sk-123
```

### 11.2 LiteLLM Unified 4000-Port Version

Use only after verifying LiteLLM fallback/routing is correct:

```bash
MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct
OPENROUTER_BASE_URL=http://127.0.0.1:4000/v1
OPENROUTER_API_KEY=sk-123

EXTRACTOR_MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct
EXTRACTOR_BASE_URL=http://127.0.0.1:4000/v1
EXTRACTOR_API_KEY=sk-123
```

---

## 12. Standard Commands

### 12.1 Image Smoke

```bash
python -u ./run_api.py --num_workers 2 --limit 5 --max_pages 5 --resolution 72 --output_path ./results/res_img_router_smoke.json 2>&1 | tee ./logs/mml_img_router_smoke.log
```

### 12.2 OCR/Text Smoke

```bash
python -u ./run_api_text.py --num_workers 2 --limit 5 --max_pages 5 --output_path ./results/res_text_router_smoke.json 2>&1 | tee ./logs/mml_text_router_smoke.log
```

### 12.3 Image Full Run

```bash
python -u ./run_api.py --num_workers 2 --max_pages 5 --resolution 72 --output_path ./results/res_img_router_full.json 2>&1 | tee ./logs/mml_img_router_full.log
```

### 12.4 OCR/Text Full Run

```bash
python -u ./run_api_text.py --num_workers 2 --max_pages 5 --output_path ./results/res_text_router_full.json 2>&1 | tee ./logs/mml_text_router_full.log
```

---

## 13. Current Code-Change Summary Relative to Upstream

### 13.1 `run_api.py`

Added:

1. OpenAI-compatible API support
2. `.env.mmlongbench` loading
3. merged resume logic
4. retryable failed-sample logic
5. route-level context fallback
6. sample-level concurrency
7. timing logs
8. route metadata logging

### 13.2 `run_api_text.py`

Added:

1. OCR/text route
2. `.env.mmlongbench` loading
3. merged resume logic
4. retryable failed-sample logic
5. route-level context fallback
6. sample-level concurrency
7. timing logs

### 13.3 `extract_answer.py`

Changed:

1. configurable client construction
2. support for custom `base_url`
3. placeholder API key for local OpenAI-compatible servers

### 13.4 `eval_score.py`

Changed:

1. safer parsing of list-like predictions
2. replaced unsafe `eval(...)` cases with guarded literal parsing

---

## 14. How to Integrate Later Baselines

This section is the most important for later expansion.

### 14.1 Principle

A new baseline should be integrated without changing downstream scoring and bookkeeping logic whenever possible.

That means:

1. Keep sample loading the same.
2. Keep output JSON schema the same.
3. Keep extraction and scoring the same where possible.
4. Replace only the "how do we produce `response`?" part.

### 14.2 Recommended Integration Pattern

Any new baseline should be organized as:

1. input construction
2. model inference
3. optional answer extraction
4. scoring
5. persistence

Current reusable components:

- sample loading and resume
- retryable failure logic
- route fallback logic
- answer extraction
- evaluation and summary writing

### 14.3 Suggested Ways to Add New Baselines

#### A. New API-Based Baseline

Example: another OpenAI-compatible multimodal model.

Implementation path:

1. duplicate `run_api.py` or factor out the request stage
2. keep:
   - `load_samples`
   - `should_skip_sample`
   - `process_one_sample` structure
   - extraction
   - scoring

Only replace:

- message / payload construction
- backend invocation

#### B. New OCR Pipeline Baseline

Example: MinerU / PaddleOCR / another structured OCR engine.

Implementation path:

1. duplicate `run_api_text.py`
2. replace `build_text_prompt(...)`
3. possibly store structured OCR artifacts separately
4. keep route / extraction / scoring logic unchanged

#### C. Retrieval-Enhanced Baseline

Example: retrieve top-k pages first, then ask model only on selected pages.

Recommended approach:

1. build a new script, e.g. `run_api_retrieval.py`
2. keep the result format identical
3. add retrieval metadata fields such as:
   - `retrieved_pages`
   - `retrieval_scores`
   - `retrieval_method`

Pipeline:

1. retrieve pages
2. construct prompt from retrieved pages
3. call model
4. extract answer
5. score

This lets you compare retrieval vs no-retrieval under the same scoring pipeline.

### 14.4 Output Schema Recommendation for Baselines

Try to preserve at least:

- `question`
- `answer`
- `answer_format`
- `response`
- `extracted_res`
- `pred`
- `score`
- `status`
- `failure_stage`
- `error`

Optional but recommended:

- `used_base_url`
- `used_model_name`
- `used_route_label`
- `used_route_max_model_len`
- `timing_prepare_seconds`
- `timing_generation_seconds`
- `timing_extraction_seconds`
- `timing_total_seconds`

For retrieval methods:

- `retrieved_pages`
- `retrieval_scores`
- `retrieval_backend`

### 14.5 Why This Structure Helps Group Discussion

For group meetings, this design makes code explanation easier because the benchmark runner can be explained as four layers:

1. **Input layer**
   - where pages/text/images come from

2. **Inference layer**
   - which model or route is used

3. **Post-processing layer**
   - how long responses become short answers

4. **Evaluation layer**
   - how benchmark scores are computed and persisted

Every new baseline only needs to replace or extend one or two of these layers.

---

## 15. Recommended Next Improvements

For future work, the most valuable improvements are:

1. add extractor fallback (`8001 -> 8002`)
2. add rule-based answer extraction before LLM extraction
3. add page-selection modes:
   - front pages
   - gold-evidence debug mode
   - retrieval-selected pages
4. add JSON-safe output backup if result JSON is corrupted
5. optionally separate preprocessing cache names by `resolution`

---

## 16. Group-Meeting Talking Points

If explaining the code in a meeting, a good concise story is:

1. The upstream benchmark originally supported a narrower set of models and a simpler save/resume flow.
2. We adapted it to OpenAI-compatible local/remote serving.
3. We added two benchmark routes:
   - image route
   - OCR/text route
4. We made execution resumable and retryable.
5. We added route-level fallback to handle context overflow across throughput / longctx / maxctx backends.
6. We added concurrency while keeping safe JSON persistence.
7. We kept the scoring layer compatible so later baselines can plug into the same evaluation pipeline.

That is the key engineering contribution of the current benchmark code.
