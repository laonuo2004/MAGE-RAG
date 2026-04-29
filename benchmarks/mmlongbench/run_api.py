import os
import re
import json
import pathlib
import sys
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter

CURRENT_DIR = pathlib.Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from tqdm import tqdm

from eval.extract_answer import build_client, extract_answer
from eval.eval_score import eval_score, eval_acc_and_f1, show_results

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.wrapper import build_context_builder

logger = logging.getLogger("mmlongbench.run_api")

MAX_TRY = 10
MAX_TOKENS = 1024
TEMPERATURE = 0.0
EXTRACTOR_PROMPT_PATH = CURRENT_DIR / "eval" / "prompt_for_answer_extraction.md"


def sample_key(sample):
    return (
        sample.get("doc_id"),
        sample.get("question"),
        sample.get("answer"),
        sample.get("answer_format"),
    )


def is_failed_response(sample):
    response = str(sample.get("response", ""))
    return response == "Failed" or response.startswith("Failed:")


def is_failed_extraction(sample):
    pred = str(sample.get("pred", ""))
    extracted_res = str(sample.get("extracted_res", ""))
    return (
        pred == "Failed to extract"
        or extracted_res == "Failed"
        or extracted_res.startswith("Failed:")
    )


def is_retryable_failure(sample):
    status = sample.get("status")
    if status in {"failed_generation", "failed_extraction"}:
        return True
    return is_failed_response(sample) or is_failed_extraction(sample)


def should_skip_sample(sample):
    return "score" in sample and not is_retryable_failure(sample)


def parse_extracted_answer(extracted_res):
    text = str(extracted_res or "")
    match = re.search(r"Extracted answer:\s*(.*?)(?:\n+Answer format:|$)", text, flags=re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def safe_load_existing_samples(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to load existing MMLongBench results from %s: %s", path, exc)
        return None


def load_samples(cfg, output_path):
    benchmark_cfg = cfg.benchmarks
    with open(benchmark_cfg.input_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    existing_samples = safe_load_existing_samples(output_path)
    if existing_samples is not None:
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
            return response.choices[0].message.content, None
        except Exception as exc:
            last_error = exc
            logger.warning(
                "MMLongBench generation failed. model=%s attempt=%s/%s error=%s",
                model_name,
                attempt,
                MAX_TRY,
                exc,
            )
    return f"Failed: {last_error}", last_error


def process_one_sample(
    sample,
    cfg,
    prompt,
    qa_model_name,
    extractor_model_name,
    base_url,
    api_key,
    context_builder=None,
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
        "Start MMLongBench sample. doc_id=%s question=%s",
        sample.get("doc_id"),
        sample.get("question"),
    )

    prep_start = perf_counter()
    if context_builder is None:
        raise ValueError("context_builder is required.")
    messages = context_builder.build("mmlongbench", sample)
    prep_seconds = perf_counter() - prep_start

    generation_start = perf_counter()
    qa_client = build_client(api_key=api_key, base_url=base_url)
    response, request_error = request_llm(messages, qa_model_name, qa_client)
    generation_seconds = perf_counter() - generation_start
    sample["response"] = response
    sample["used_base_url"] = base_url
    sample["used_model_name"] = qa_model_name
    sample["used_route_label"] = "litellm"
    sample["used_route_max_model_len"] = None
    sample["timing_prepare_seconds"] = round(prep_seconds, 3)
    sample["timing_generation_seconds"] = round(generation_seconds, 3)

    if is_failed_response(sample):
        sample["error"] = repr(request_error) if request_error is not None else sample.get("error")
        sample["failure_stage"] = "generation"
        sample["extracted_res"] = "Failed"
        sample["pred"] = "Failed to extract"
        sample["score"] = 0.0
        sample["status"] = "failed_generation"
        sample["timing_extraction_seconds"] = 0.0
        sample["timing_total_seconds"] = round(perf_counter() - total_start, 3)
        return sample

    extractor_client = build_client(
        api_key=api_key,
        base_url=base_url,
    )
    extraction_start = perf_counter()
    extracted_res = extract_answer(
        sample["question"],
        response,
        prompt,
        model_name=extractor_model_name,
        client=extractor_client,
    )
    extraction_seconds = perf_counter() - extraction_start
    sample["extractor_model_name"] = extractor_model_name
    sample["extracted_res"] = extracted_res
    pred_ans = parse_extracted_answer(extracted_res)
    if pred_ans is None:
        sample["pred"] = "Failed to extract"
        sample["score"] = 0.0
        sample["status"] = "failed_extraction"
        sample["failure_stage"] = "extraction"
    else:
        sample["pred"] = pred_ans
        sample["score"] = eval_score(sample["answer"], pred_ans, sample["answer_format"])
        sample["status"] = "completed"
    sample["timing_extraction_seconds"] = round(extraction_seconds, 3)
    sample["timing_total_seconds"] = round(perf_counter() - total_start, 3)
    return sample


def build_default_results_file(cfg, benchmark_cfg):
    baseline_name = cfg.baselines.name
    model_name = benchmark_cfg.qa_model_name.replace("/", "_").replace(":free", "").replace("-", "_")
    return os.path.join(benchmark_cfg.results_dir, f"res_{baseline_name}_{model_name}.json")


def write_results(samples, output_path):
    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False)
    os.replace(tmp_path, output_path)


def log_sample_result(sample, samples):
    acc, f1 = eval_acc_and_f1(samples)
    logger.info("--------------------------------------")
    logger.info("Question: %s", sample["question"])
    logger.info("LiteLLM: %s | %s", sample.get("used_base_url"), sample.get("used_model_name"))
    logger.info(
        "Timing: Prepare=%.3fs | Generation=%.3fs | Extraction=%.3fs | Total=%.3fs",
        sample.get("timing_prepare_seconds", 0.0),
        sample.get("timing_generation_seconds", 0.0),
        sample.get("timing_extraction_seconds", 0.0),
        sample.get("timing_total_seconds", 0.0),
    )
    logger.debug("Response: %s", sample["response"])
    logger.info("GT: %s\tPred: %s\tScore: %s", sample["answer"], sample["pred"], sample["score"])
    logger.info("Avg Acc: %s", acc)
    logger.info("Avg F1: %s", f1)


def evaluate(cfg, samples, output_path, prompt, context_builder):
    benchmark_cfg = cfg.benchmarks
    process_mode = benchmark_cfg.process_mode
    workers = int(benchmark_cfg.workers)
    qa_model_name = benchmark_cfg.qa_model_name
    extractor_model_name = benchmark_cfg.extractor_model_name
    base_url = cfg.litellm.base_url
    api_key = cfg.litellm.api_key

    completed_count = sum(1 for s in samples if should_skip_sample(s))
    total_count = len(samples)
    pending_indices = [idx for idx, sample in enumerate(samples) if not should_skip_sample(sample)]

    logger.info("Progress: %s/%s completed", completed_count, total_count)
    logger.info("Evaluation Process Mode: %s", process_mode)
    if process_mode == "parallel":
        logger.info("Number Of Worker Threads: %s", workers)

    if not pending_indices:
        logger.info("No Data To Process.")
        return

    def run_index(idx):
        return process_one_sample(
            samples[idx],
            cfg,
            prompt,
            qa_model_name,
            extractor_model_name,
            base_url,
            api_key,
            context_builder=context_builder,
        )

    if process_mode == "serial":
        for idx in tqdm(pending_indices, desc="Processing"):
            sample = run_index(idx)
            samples[idx] = sample
            log_sample_result(sample, samples)
            write_results(samples, output_path)
        return

    if process_mode == "parallel":
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            future_to_index = {executor.submit(run_index, idx): idx for idx in pending_indices}
            for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Processing"):
                idx = future_to_index[future]
                sample = future.result()
                samples[idx] = sample
                log_sample_result(sample, samples)
                write_results(samples, output_path)
        return

    raise ValueError(f"Unsupported process_mode: {process_mode}")


def run_mmlongbench(cfg):
    benchmark_cfg = cfg.benchmarks
    qa_model_name = benchmark_cfg.qa_model_name
    extractor_model_name = benchmark_cfg.extractor_model_name
    base_url = cfg.litellm.base_url

    if not qa_model_name:
        raise ValueError("Missing model name. Set benchmarks.qa_model_name in the Hydra config.")
    if not extractor_model_name:
        raise ValueError("Missing extractor model name. Set benchmarks.extractor_model_name in the Hydra config.")
    if not base_url:
        raise ValueError("Missing LiteLLM base URL. Set litellm.base_url in the Hydra config.")

    output_path = benchmark_cfg.results_file or build_default_results_file(cfg, benchmark_cfg)
    os.makedirs(benchmark_cfg.results_dir, exist_ok=True)
    os.makedirs(benchmark_cfg.tmp_dir, exist_ok=True)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logger.info("Output Datapath: %s", output_path)
    logger.info("LiteLLM Base URL: %s", base_url)
    logger.info("QA Model: %s", qa_model_name)
    logger.info("Extractor Model: %s", extractor_model_name)

    with open(EXTRACTOR_PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt = f.read()
    samples = load_samples(cfg, output_path)
    context_builder = build_context_builder(cfg)

    evaluate(cfg, samples, output_path, prompt, context_builder)

    show_results(samples, show_path=re.sub(r"\.json$", ".txt", output_path))
    return samples
