import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter

import fitz
from tqdm import tqdm

from env_utils import get_config_value, load_local_env
from eval.extract_answer import build_client, extract_answer
from eval.eval_score import eval_acc_and_f1, eval_score, show_results
from route_utils import build_routes, find_next_route_index, is_context_overflow_error, safe_load_existing_samples


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


def slugify_filename(text, max_len=80):
    text = re.sub(r"[^0-9a-zA-Z._-]+", "_", str(text))
    return text[:max_len].strip("_") or "sample"


def build_text_prompt(sample, args):
    question = sample["question"]
    pdf_path = os.path.join(args.document_path, sample["doc_id"])

    page_blocks = []
    with fitz.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf[: args.max_pages], start=1):
            page_text = page.get_text("text").strip()
            if not page_text:
                page_text = "[EMPTY PAGE]"
            page_blocks.append(f"[Page {page_idx}]\n{page_text}")

    document_text = "\n\n".join(page_blocks)
    prompt = (
        "You are given the OCR/text extracted from a long PDF document.\n"
        "Answer the question using only the provided document text.\n"
        "If the answer cannot be found, say Not answerable.\n\n"
        f"Question:\n{question}\n\n"
        f"Document text:\n{document_text}"
    )
    return [{"role": "user", "content": prompt}]


def load_samples(args):
    with open(args.input_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if args.limit is not None:
        samples = samples[: args.limit]

    existing_samples = safe_load_existing_samples(args.output_path)
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


def ensure_debug_dir(args):
    if not args.debug_prompts:
        return None
    debug_dir = args.debug_dir or "./debug_prompts"
    os.makedirs(debug_dir, exist_ok=True)
    return debug_dir


def write_debug_record(args, sample, payload):
    debug_dir = ensure_debug_dir(args)
    if debug_dir is None:
        return

    sample_id = sample.get("sample_id")
    if sample_id is None:
        sample_id = f"{sample.get('doc_id', 'doc')}__{slugify_filename(sample.get('question', 'question'), 60)}"

    path = os.path.join(debug_dir, f"{slugify_filename(sample_id, 120)}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def request_with_fallback(messages, args, routes):
    route_index = 0
    last_error = None
    while route_index is not None and route_index < len(routes):
        route = routes[route_index]
        client = build_client(api_key=route.api_key, base_url=route.base_url)
        attempts = 0
        while attempts < args.max_try:
            try:
                completion = client.chat.completions.create(
                    model=route.model_name,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                return completion.choices[0].message.content, route, None
            except Exception as exc:
                attempts += 1
                last_error = exc
                message = str(exc)
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
            route_index = next_route_index
            continue
    return f"Failed: {last_error}", routes[min(route_index or 0, len(routes) - 1)], last_error


def process_one_sample(sample, args, prompt, routes):
    total_start = perf_counter()
    sample = dict(sample)
    sample.pop("score", None)
    sample.pop("pred", None)
    sample.pop("extracted_res", None)
    sample.pop("error", None)
    sample.pop("failure_stage", None)
    sample.pop("status", None)

    prep_start = perf_counter()
    messages = build_text_prompt(sample, args)
    prep_seconds = perf_counter() - prep_start
    debug_payload = {
        "question": sample.get("question"),
        "doc_id": sample.get("doc_id"),
        "answer": sample.get("answer"),
        "answer_format": sample.get("answer_format"),
        "evidence_pages": sample.get("evidence_pages"),
        "evidence_sources": sample.get("evidence_sources"),
        "input_messages": messages,
    }

    generation_start = perf_counter()
    response, used_route, request_error = request_with_fallback(messages, args, routes)
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
        write_debug_record(args, sample, debug_payload)
        return sample

    extractor_model_name = args.extractor_model_name or used_route.model_name
    extractor_client = build_client(
        api_key=args.extractor_api_key if args.extractor_model_name else used_route.api_key,
        base_url=args.extractor_base_url if args.extractor_model_name else used_route.base_url,
    )
    extractor_input = {
        "model_name": extractor_model_name,
        "base_url": args.extractor_base_url if args.extractor_model_name else used_route.base_url,
        "question": sample["question"],
        "prompt_template": prompt,
        "analysis": response,
    }
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
    try:
        if pred_ans is None:
            raise ValueError("failed to parse extracted answer")
        score = eval_score(sample["answer"], pred_ans, sample["answer_format"])
        sample["status"] = "completed"
    except Exception:
        pred_ans = "Failed to extract"
        score = 0.0
        sample["status"] = "failed_extraction"
        sample["failure_stage"] = "extraction"
    sample["pred"] = pred_ans
    sample["score"] = score
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
    write_debug_record(args, sample, debug_payload)
    return sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./data/samples.json")
    parser.add_argument("--document_path", type=str, default="./data/documents")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--extractor_model_name", type=str, default=None)
    parser.add_argument("--extractor_base_url", type=str, default=None)
    parser.add_argument("--extractor_api_key", type=str, default=None)
    parser.add_argument("--max_pages", type=int, default=120)
    parser.add_argument("--max_try", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--extractor_prompt_path", type=str, default="./eval/prompt_for_answer_extraction.md")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--route_base_urls", type=str, default=None)
    parser.add_argument("--route_model_names", type=str, default=None)
    parser.add_argument("--route_api_keys", type=str, default=None)
    parser.add_argument("--route_labels", type=str, default=None)
    parser.add_argument("--route_max_model_lens", type=str, default=None)
    parser.add_argument("--debug_prompts", action="store_true")
    parser.add_argument("--debug_dir", type=str, default=None)
    args = parser.parse_args()
    local_env = load_local_env()

    args.model_name = get_config_value(args.model_name, "MODEL_NAME", local_env=local_env)
    args.base_url = get_config_value(args.base_url, "OPENROUTER_BASE_URL", local_env=local_env)
    args.api_key = get_config_value(args.api_key, "OPENROUTER_API_KEY", local_env=local_env)
    args.extractor_model_name = get_config_value(
        args.extractor_model_name,
        "EXTRACTOR_MODEL_NAME",
        local_env=local_env,
        default=args.model_name,
    )
    args.extractor_base_url = get_config_value(
        args.extractor_base_url,
        "EXTRACTOR_BASE_URL",
        local_env=local_env,
        default=args.base_url,
    )
    args.extractor_api_key = get_config_value(
        args.extractor_api_key,
        "EXTRACTOR_API_KEY",
        local_env=local_env,
        default=args.api_key,
    )
    args.route_base_urls = get_config_value(args.route_base_urls, "ROUTE_BASE_URLS", local_env=local_env)
    args.route_model_names = get_config_value(args.route_model_names, "ROUTE_MODEL_NAMES", local_env=local_env)
    args.route_api_keys = get_config_value(args.route_api_keys, "ROUTE_API_KEYS", local_env=local_env)
    args.route_labels = get_config_value(args.route_labels, "ROUTE_LABELS", local_env=local_env)
    args.route_max_model_lens = get_config_value(args.route_max_model_lens, "ROUTE_MAX_MODEL_LENS", local_env=local_env)

    if not args.model_name:
        raise ValueError("Missing model name. Set --model_name or MODEL_NAME in .env.mmlongbench/.env.")

    model_slug = re.sub(r"[^0-9a-zA-Z._-]+", "_", args.model_name)
    args.output_path = args.output_path or f"./results/res_text_{model_slug}.json"
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./tmp", exist_ok=True)
    if args.debug_prompts:
        args.num_workers = 1
        ensure_debug_dir(args)

    routes = build_routes(
        model_name=args.model_name,
        base_url=args.base_url,
        api_key=args.api_key,
        route_model_names=args.route_model_names,
        route_base_urls=args.route_base_urls,
        route_api_keys=args.route_api_keys,
        route_labels=args.route_labels,
        route_max_model_lens=args.route_max_model_lens,
    )

    with open(args.extractor_prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()

    samples = load_samples(args)
    completed_count = sum(1 for s in samples if should_skip_sample(s))
    total_count = len(samples)
    print(f"Progress: {completed_count}/{total_count} completed")
    pending_indices = [idx for idx, sample in enumerate(samples) if not should_skip_sample(sample)]
    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
        future_to_index = {
            executor.submit(process_one_sample, samples[idx], args, prompt, routes): idx for idx in pending_indices
        }
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Processing OCR/Text route"):
            idx = future_to_index[future]
            sample = future.result()
            samples[idx] = sample
            acc, f1 = eval_acc_and_f1(samples)
            print("--------------------------------------")
            print("Question: {}".format(sample["question"]))
            print("Route: {} | {} | {}".format(sample.get("used_route_label"), sample.get("used_base_url"), sample.get("used_model_name")))
            print(
                "Timing: prepare={:.3f}s | generation={:.3f}s | extraction={:.3f}s | total={:.3f}s".format(
                    sample.get("timing_prepare_seconds", 0.0),
                    sample.get("timing_generation_seconds", 0.0),
                    sample.get("timing_extraction_seconds", 0.0),
                    sample.get("timing_total_seconds", 0.0),
                )
            )
            print("Response: {}".format(sample["response"]))
            print("Gt: {}\tPred: {}\tScore: {}".format(sample["answer"], sample["pred"], sample["score"]))
            print("Avg acc: {}".format(acc))
            print("Avg f1: {}".format(f1))
            with open(args.output_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, ensure_ascii=False)

    show_results(samples, show_path=re.sub(r"\.json$", ".txt", args.output_path))
