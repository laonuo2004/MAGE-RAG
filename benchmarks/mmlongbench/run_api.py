import os
import re
import json
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter

CURRENT_DIR = pathlib.Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from tqdm import tqdm

from eval.extract_answer import build_client, extract_answer
from eval.eval_score import eval_score, eval_acc_and_f1, show_results
from route_utils import build_routes, find_next_route_index, is_context_overflow_error, safe_load_existing_samples

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.wrapper import build_context_builder

MAX_TRY = 10
MAX_TOKENS = 1024
TEMPERATURE = 0.0
EXTRACTOR_PROMPT_PATH = CURRENT_DIR / "eval" / "prompt_for_answer_extraction.md"


def _resolve_provider_cfg(cfg, benchmark_cfg, default_provider="litellm"):
    provider_name = benchmark_cfg.get("llm_provider", default_provider)
    providers = cfg.get("llm_providers", {}) or {}
    provider_cfg = providers.get(provider_name, {}) or {}
    return provider_name, provider_cfg


def _resolve_model_name(model_name, provider_cfg):
    if model_name is None:
        return None
    model_mapping = provider_cfg.get("model_mapping", {}) or {}
    return model_mapping.get(model_name, model_name)


def resolve_model_name(cfg, model_name=None):
    benchmark_cfg = cfg.benchmarks
    _, provider_cfg = _resolve_provider_cfg(cfg, benchmark_cfg)
    return _resolve_model_name(model_name or benchmark_cfg.model_name, provider_cfg)


def resolve_base_url(cfg):
    benchmark_cfg = cfg.benchmarks
    _, provider_cfg = _resolve_provider_cfg(cfg, benchmark_cfg)
    return provider_cfg.get("base_url")


def resolve_api_key(cfg):
    benchmark_cfg = cfg.benchmarks
    _, provider_cfg = _resolve_provider_cfg(cfg, benchmark_cfg)
    return provider_cfg.get("api_key")


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


def request_with_fallback(messages, routes):
    route_index = 0
    last_error = None
    while route_index is not None and route_index < len(routes):
        route = routes[route_index]
        print(
            "[ROUTE TRY] Label={} | Base URL={} | Model={} | Max Model Len={}".format(
                route.label,
                route.base_url,
                route.model_name,
                route.max_model_len,
            )
        )
        client = build_client(api_key=route.api_key, base_url=route.base_url)
        attempts = 0
        while attempts < MAX_TRY:
            try:
                response = client.chat.completions.create(
                    model=route.model_name,
                    messages=messages,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                )
                return response.choices[0].message.content, route, None
            except Exception as exc:
                attempts += 1
                last_error = exc
                message = str(exc)
                print(
                    "[ROUTE ERROR] Label={} | Attempt={}/{} | Error={}".format(
                        route.label,
                        attempts,
                        MAX_TRY,
                        exc,
                    )
                )
                if is_context_overflow_error(message):
                    next_route_index = find_next_route_index(routes, route_index, message)
                    if next_route_index is None:
                        return f"Failed: {exc}", route, exc
                    route_index = next_route_index
                    break
        else:
            next_route_index = route_index + 1 if route_index + 1 < len(routes) else None
            if next_route_index is None:
                return f"Failed: {last_error}", route, last_error
            print(
                "[ROUTE FALLBACK] From={} -> To={}".format(
                    route.label,
                    routes[next_route_index].label,
                )
            )
            route_index = next_route_index
            continue
    return f"Failed: {last_error}", routes[min(route_index or 0, len(routes) - 1)], last_error


def process_one_sample(sample, cfg, prompt, routes, context_builder=None):
    total_start = perf_counter()
    sample = dict(sample)
    sample.pop("score", None)
    sample.pop("pred", None)
    sample.pop("extracted_res", None)
    sample.pop("error", None)
    sample.pop("failure_stage", None)
    sample.pop("status", None)

    print(
        "[START] Doc ID={} | Question={}".format(
            sample.get("doc_id"),
            sample.get("question"),
        )
    )

    print("[STAGE] Prepare")
    prep_start = perf_counter()
    if context_builder is None:
        raise ValueError("context_builder is required.")
    messages = context_builder.build("mmlongbench", sample)
    metadata = getattr(messages, "metadata", {})
    prep_seconds = perf_counter() - prep_start
    debug_payload = {
        "question": sample.get("question"),
        "doc_id": sample.get("doc_id"),
        "answer": sample.get("answer"),
        "answer_format": sample.get("answer_format"),
        "evidence_pages": sample.get("evidence_pages"),
        "evidence_sources": sample.get("evidence_sources"),
        "input_messages": messages,
        "context_metadata": metadata,
    }

    print("[STAGE] Generation")
    generation_start = perf_counter()
    response, used_route, request_error = request_with_fallback(messages, routes)
    generation_seconds = perf_counter() - generation_start
    sample["response"] = response
    sample["used_base_url"] = used_route.base_url
    sample["used_model_name"] = used_route.model_name
    sample["used_route_label"] = used_route.label
    sample["used_route_max_model_len"] = used_route.max_model_len
    sample["timing_prepare_seconds"] = round(prep_seconds, 3)
    sample["timing_generation_seconds"] = round(generation_seconds, 3)
    debug_payload["used_route"] = {
        "label": used_route.label,
        "base_url": used_route.base_url,
        "model_name": used_route.model_name,
        "max_model_len": used_route.max_model_len,
    }
    debug_payload["response"] = response

    if is_failed_response(sample):
        sample["error"] = repr(request_error) if request_error is not None else sample.get("error")
        sample["failure_stage"] = "generation"
        sample["extracted_res"] = "Failed"
        sample["pred"] = "Failed to extract"
        sample["score"] = 0.0
        sample["status"] = "failed_generation"
        sample["timing_extraction_seconds"] = 0.0
        sample["timing_total_seconds"] = round(perf_counter() - total_start, 3)
        debug_payload["request_error"] = repr(request_error) if request_error is not None else None
        debug_payload["extractor"] = None
        debug_payload["timing"] = {
            "prepare_seconds": sample["timing_prepare_seconds"],
            "generation_seconds": sample["timing_generation_seconds"],
            "extraction_seconds": sample["timing_extraction_seconds"],
            "total_seconds": sample["timing_total_seconds"],
        }
        return sample

    extractor_model_name = used_route.model_name
    extractor_base_url = used_route.base_url
    extractor_api_key = used_route.api_key
    extractor_client = build_client(
        api_key=extractor_api_key,
        base_url=extractor_base_url,
    )
    extractor_input = {
        "model_name": extractor_model_name,
        "base_url": extractor_base_url,
        "question": sample["question"],
        "prompt_template": prompt,
        "analysis": response,
    }
    print("[STAGE] Extraction")
    extraction_start = perf_counter()
    extracted_res = extract_answer(
        sample["question"],
        response,
        prompt,
        model_name=extractor_model_name,
        client=extractor_client,
    )
    extraction_seconds = perf_counter() - extraction_start
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
    debug_payload["extractor"] = extractor_input
    debug_payload["extracted_res"] = extracted_res
    debug_payload["pred"] = sample["pred"]
    debug_payload["status"] = sample["status"]
    debug_payload["timing"] = {
        "prepare_seconds": sample["timing_prepare_seconds"],
        "generation_seconds": sample["timing_generation_seconds"],
        "extraction_seconds": sample["timing_extraction_seconds"],
        "total_seconds": sample["timing_total_seconds"],
    }
    return sample


def run_mmlongbench(cfg):
    benchmark_cfg = cfg.benchmarks
    context_builder_name = cfg.baselines.name
    model_name = resolve_model_name(cfg)
    base_url = resolve_base_url(cfg)
    api_key = resolve_api_key(cfg)

    if not model_name:
        raise ValueError("Missing model name. Set benchmarks.model_name in the Hydra config.")

    model_slug = re.sub(r"[^0-9a-zA-Z._-]+", "_", model_name)
    baseline_slug = re.sub(r"[^0-9a-zA-Z._-]+", "_", context_builder_name)
    output_path = os.path.join(benchmark_cfg.results_dir, f"res_{baseline_slug}_{model_slug}.json")
    os.makedirs(benchmark_cfg.results_dir, exist_ok=True)
    os.makedirs(benchmark_cfg.tmp_dir, exist_ok=True)

    routes = build_routes(
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
    )

    with open(EXTRACTOR_PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt = f.read()
    samples = load_samples(cfg, output_path)
    context_builder = build_context_builder(cfg)
    
    # 计算已完成和待完成样本
    completed_count = sum(1 for s in samples if should_skip_sample(s))
    total_count = len(samples)
    print(f"Progress: {completed_count}/{total_count} Completed")
    pending_indices = [idx for idx, sample in enumerate(samples) if not should_skip_sample(sample)]

    with ThreadPoolExecutor(max_workers=max(1, benchmark_cfg.num_workers)) as executor:
        future_to_index = {
            executor.submit(process_one_sample, samples[idx], cfg, prompt, routes, context_builder): idx for idx in pending_indices
        }
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Processing"):
            idx = future_to_index[future]
            sample = future.result()
            samples[idx] = sample
            acc, f1 = eval_acc_and_f1(samples)
            print("--------------------------------------")
            print("Question: {}".format(sample["question"]))
            print("Route: {} | {} | {}".format(sample.get("used_route_label"), sample.get("used_base_url"), sample.get("used_model_name")))
            print(
                "Timing: Prepare={:.3f}s | Generation={:.3f}s | Extraction={:.3f}s | Total={:.3f}s".format(
                    sample.get("timing_prepare_seconds", 0.0),
                    sample.get("timing_generation_seconds", 0.0),
                    sample.get("timing_extraction_seconds", 0.0),
                    sample.get("timing_total_seconds", 0.0),
                )
            )
            print("Response: {}".format(sample["response"]))
            print("GT: {}\tPred: {}\tScore: {}".format(sample["answer"], sample["pred"], sample["score"]))
            print("Avg Acc: {}".format(acc))
            print("Avg F1: {}".format(f1))
            with open(output_path, 'w', encoding="utf-8") as f:
                json.dump(samples, f, ensure_ascii=False)
    
    show_results(samples, show_path=re.sub(r"\.json$", ".txt", output_path))
    return samples
