import os
import re
import json
import pathlib
import sys
import logging
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter

CURRENT_DIR = pathlib.Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from tqdm import tqdm

from eval.extract_answer import extract_answer
from eval.eval_score import eval_score, eval_acc_and_f1, show_results
from utils.logging_utils import apply_logging_config

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.wrapper import build_context_builder

logger = logging.getLogger("mmlongbench.run_api")

MAX_TRY = 10
MAX_TOKENS = 1024
TEMPERATURE = 0.0
MAX_RETRY_ROUNDS = 3
EXTRACTOR_PROMPT_PATH = CURRENT_DIR / "eval" / "prompt_for_answer_extraction.md"


def sample_key(sample):
    return (
        sample.get("doc_id"),
        sample.get("question"),
        sample.get("answer"),
        sample.get("answer_format"),
    )


def should_skip_sample(sample):
    return "score" in sample and sample.get("pred") != "Failed to extract"


def parse_extracted_answer(extracted_res):
    text = str(extracted_res or "")
    match = re.search(r"Extracted answer:\s*(.*?)(?:\n+Answer format:|$)", text, flags=re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def read_jsonl_results(path):
    results = []
    if not os.path.exists(path):
        return results
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    result = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Skipping invalid JSONL line in %s:%s: %s",
                        path,
                        line_no,
                        exc,
                    )
                    continue
                if should_skip_sample(result):
                    results.append(result)
    except Exception as exc:
        logger.warning("Failed to load existing MMLongBench results from %s: %s", path, exc)
    return results


def load_samples(cfg, output_path):
    benchmark_cfg = cfg.benchmarks
    with open(benchmark_cfg.input_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    logger.info("Loaded %s samples from %s", len(samples), benchmark_cfg.input_path)
    
    existing_samples = read_jsonl_results(output_path)
    if existing_samples:
        existing_by_key = {sample_key(sample): sample for sample in existing_samples}
        merged_samples = []
        for sample in samples:
            existing = existing_by_key.get(sample_key(sample))
            if existing is not None:
                merged = dict(sample)
                merged.update(existing)
                merged_samples.append(merged)
            else:
                merged_samples.append(sample)
        
        logger.info("Merged existing results with loaded samples. Total samples after merge: %s", len(merged_samples))
        return merged_samples

    return samples


def request_llm(messages, model_name, client):
    last_error = None
    for attempt in range(1, MAX_TRY + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            logger.debug("Raw LLM response: %s", response.choices[0].message)
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("LLM response content is None")
            return content
        except Exception as exc:
            last_error = exc
            logger.warning(
                "MMLongBench generation failed. model=%s attempt=%s/%s error=%s",
                model_name,
                attempt,
                MAX_TRY,
                exc,
            )
    return f"Failed: {last_error}"


def process_one_sample(
    sample,
    cfg,
    prompt
):
    total_start = perf_counter()
    sample = dict(sample)
    sample.pop("score", None)
    sample.pop("pred", None)
    sample.pop("extracted_res", None)
    sample.pop("error", None)
    sample.pop("failure_stage", None)
    sample.pop("status", None)

    logger.debug(
        "doc_id=%s question=%s",
        sample.get("doc_id"),
        sample.get("question"),
    )

    prep_start = perf_counter()
    # ========== 这一部分抽象程度较高，需要仔细分析理解 ==========
    context_builder = build_context_builder(cfg)
    messages = context_builder.build("mmlongbench", sample)
    # =========================================================
    benchmark_cfg = cfg.benchmarks
    client = OpenAI(api_key=cfg.litellm.api_key, base_url=cfg.litellm.base_url)
    qa_model_name = benchmark_cfg.qa_model_name
    extractor_model_name = benchmark_cfg.extractor_model_name
    prep_seconds = perf_counter() - prep_start

    generation_start = perf_counter()
    response = request_llm(messages, qa_model_name, client)
    generation_seconds = perf_counter() - generation_start
    logger.debug(f"Time taken for LLM to answer question_id {sample.get('question_id', 'unknown')}: {generation_seconds:.2f} seconds")
    sample["response"] = response
    sample["timing_prepare_seconds"] = round(prep_seconds, 3)
    sample["timing_generation_seconds"] = round(generation_seconds, 3)

    extraction_start = perf_counter()
    extracted_res = extract_answer(
        sample["question"],
        response,
        prompt,
        model_name=extractor_model_name,
        client=client,
    )
    extraction_seconds = perf_counter() - extraction_start
    logger.debug(f"Time taken for Extractor LLM to answer question_id {sample.get('question_id', 'unknown')}: {extraction_seconds:.2f} seconds")
    sample["extractor_model_name"] = extractor_model_name
    sample["extracted_res"] = extracted_res
    pred_ans = parse_extracted_answer(extracted_res)
    if pred_ans is None:
        logger.warning(
            "Failed to extract answer. doc_id=%s question_id=%s",
            sample.get("doc_id"),
            sample.get("question_id", "unknown"),
        )
        return None
    else:
        sample["pred"] = pred_ans
        sample["score"] = eval_score(sample["answer"], pred_ans, sample["answer_format"])
    sample["timing_extraction_seconds"] = round(extraction_seconds, 3)
    sample["timing_total_seconds"] = round(perf_counter() - total_start, 3)
    return sample


def build_default_results_file(cfg, benchmark_cfg):
    baseline_name = cfg.baselines.name
    model_name = benchmark_cfg.qa_model_name.replace("/", "_").replace(":free", "").replace("-", "_")
    return os.path.join(benchmark_cfg.results_dir, f"res_{baseline_name}_{model_name}.jsonl")


def append_result(sample, output_path):
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def compact_results_file(output_path):
    if not os.path.exists(output_path):
        return
    raw_count = 0
    unique_results = {}
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_count += 1
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            if should_skip_sample(sample):
                unique_results[sample_key(sample)] = sample
    if len(unique_results) == raw_count:
        return
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in unique_results.values():
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def handle_sample_result(sample, samples, idx, output_path):
    if sample is None:
        logger.warning(
            "Skipping failed MMLongBench sample without writing JSONL. idx=%s doc_id=%s question_id=%s",
            idx,
            samples[idx].get("doc_id"),
            samples[idx].get("question_id", "unknown"),
        )
        return
    samples[idx] = sample
    log_sample_result(sample, samples)
    append_result(sample, output_path)


def log_pending_samples(samples):
    for idx, sample in enumerate(samples):
        if should_skip_sample(sample):
            continue
        logger.warning(
            "Pending MMLongBench sample after retries. idx=%s doc_id=%s question_id=%s question=%s",
            idx,
            sample.get("doc_id"),
            sample.get("question_id", "unknown"),
            sample.get("question"),
        )


def log_sample_result(sample, samples):
    acc, f1 = eval_acc_and_f1(samples)
    logger.debug("Question: %s", sample["question"])
    logger.debug("LiteLLM: %s | %s", sample.get("used_base_url"), sample.get("used_model_name"))
    logger.debug(
        "Timing: Prepare=%.3fs | Generation=%.3fs | Extraction=%.3fs | Total=%.3fs",
        sample.get("timing_prepare_seconds", 0.0),
        sample.get("timing_generation_seconds", 0.0),
        sample.get("timing_extraction_seconds", 0.0),
        sample.get("timing_total_seconds", 0.0),
    )
    logger.debug("Response: %s", sample.get("response"))
    logger.debug("GT: %s\tPred: %s\tScore: %s", sample.get("answer"), sample.get("pred"), sample.get("score"))
    logger.debug("Avg Acc: %s", acc)
    logger.debug("Avg F1: %s", f1)


def evaluate(cfg, samples, output_path):
    benchmark_cfg = cfg.benchmarks
    process_mode = benchmark_cfg.process_mode
    workers = int(benchmark_cfg.workers)

    completed_count = sum(1 for s in samples if should_skip_sample(s))
    total_count = len(samples)
    pending_indices = [idx for idx, sample in enumerate(samples) if not should_skip_sample(sample)]

    with open(EXTRACTOR_PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt = f.read()

    logger.info("Progress: %s/%s completed", completed_count, total_count)
    logger.info("Evaluation Process Mode: %s", process_mode)
    if process_mode == "parallel":
        logger.info("Number Of Worker Threads: %s", workers)

    if not pending_indices:
        logger.info("No Data To Process.")
        return

    def run_index(idx):
        try:
            return process_one_sample(
                samples[idx],
                cfg,
                prompt
            )
        except Exception as exc:
            logger.exception(
                "MMLongBench sample failed. idx=%s doc_id=%s question_id=%s",
                idx,
                samples[idx].get("doc_id"),
                samples[idx].get("question_id", "unknown"),
            )
            return None

    if process_mode == "serial":
        for idx in pending_indices:
            sample = run_index(idx)
            handle_sample_result(sample, samples, idx, output_path)
        return

    if process_mode == "parallel":
                        
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            future_to_index = {executor.submit(run_index, idx): idx for idx in pending_indices}
            for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Processing", mininterval=1):
                idx = future_to_index[future]
                sample = future.result()
                handle_sample_result(sample, samples, idx, output_path)
        return

    raise ValueError(f"Unsupported process_mode: {process_mode}")


def run_mmlongbench(cfg):
    apply_logging_config(cfg)

    benchmark_cfg = cfg.benchmarks
    output_path = benchmark_cfg.results_file or build_default_results_file(cfg, benchmark_cfg)
    os.makedirs(benchmark_cfg.results_dir, exist_ok=True)
    os.makedirs(benchmark_cfg.tmp_dir, exist_ok=True)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logger.info("Output Datapath: %s", output_path)
    compact_results_file(output_path)

    samples = []
    for round_id in range(1, MAX_RETRY_ROUNDS + 1):
        samples = load_samples(cfg, output_path)
        completed_count = sum(1 for sample in samples if should_skip_sample(sample))
        if completed_count == len(samples):
            break
        logger.info(
            "MMLongBench retry round %s: %s/%s successful samples found in JSONL",
            round_id,
            completed_count,
            len(samples),
        )
        evaluate(cfg, samples, output_path)

    samples = load_samples(cfg, output_path)
    completed_count = sum(1 for sample in samples if should_skip_sample(sample))
    if completed_count < len(samples):
        logger.warning(
            "MMLongBench stopped after %s retry rounds with %s/%s successful samples.",
            MAX_RETRY_ROUNDS,
            completed_count,
            len(samples),
        )
        log_pending_samples(samples)

    show_results(samples, show_path=str(pathlib.Path(output_path).with_suffix(".txt")))
    return samples
