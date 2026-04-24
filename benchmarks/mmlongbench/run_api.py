import os
import re
import json
import base64
import argparse
import fitz
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from PIL import Image
from tqdm import tqdm

from env_utils import get_config_value, load_local_env
from eval.extract_answer import build_client, extract_answer
from eval.eval_score import eval_score, eval_acc_and_f1, show_results
from route_utils import build_routes, find_next_route_index, is_context_overflow_error, safe_load_existing_samples


cached_image_list = dict()
doc_locks = {}
doc_locks_guard = Lock()


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


def get_doc_lock(doc_id):
    with doc_locks_guard:
        if doc_id not in doc_locks:
            doc_locks[doc_id] = Lock()
        return doc_locks[doc_id]


def encode_image_to_base64(img):
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def process_sample_openai_compatible(sample, args):
    question = sample["question"]
    doc_name = re.sub(r"\.pdf$", "", sample["doc_id"]).split("/")[-1]

    image_list = list()
    with get_doc_lock(sample["doc_id"]):
        with fitz.open(os.path.join(args.document_path, sample["doc_id"])) as pdf:
            for index, page in enumerate(pdf[:args.max_pages]):
                if not os.path.exists(f"./tmp/{doc_name}_{index+1}.png"):
                    image = page.get_pixmap(dpi=args.resolution)
                    image.save(f"./tmp/{doc_name}_{index+1}.png")
                image = Image.open(f"./tmp/{doc_name}_{index+1}.png")
                encoded_image = encode_image_to_base64(image)
                image_list.append(encoded_image)

    content = list()
    content.append(
        {
            "type": "text",
            "text": question,
        }
    )
    for encoded_image in image_list:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
        })
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    return messages


def process_sample_gemini(sample, args, mode):
    question = sample["question"]
    doc_name = re.sub(r"\.pdf$", "", sample["doc_id"]).split("/")[-1]

    image_list = list()
    with fitz.open(os.path.join(args.document_path, sample["doc_id"])) as pdf:
        if mode=="png":
            for index, page in enumerate(pdf[:args.max_pages]):
                if not os.path.exists(f"./tmp/{doc_name}_{index+1}.png"):
                    im = page.get_pixmap(dpi=args.resolution)
                    im.save(f"./tmp/{doc_name}_{index+1}.png")
                image_list.append(Image.open(f"./tmp/{doc_name}_{index+1}.png"))
        else:
            if sample["doc_id"] in cached_image_list:
                image_list = cached_image_list[sample["doc_id"]]
            else:
                for index, page in enumerate(pdf[:args.max_pages]):
                    if not os.path.exists(f"./tmp/{doc_name}_{index+1}.png"):
                        im = page.get_pixmap(dpi=args.resolution)
                        im.save(f"./tmp/{doc_name}_{index+1}.png")
                    image_list.append(genai.upload_file(f"./tmp/{doc_name}_{index+1}.png"))
                cached_image_list[sample["doc_id"]] = image_list
    
    return [question] + image_list


def process_sample(sample, args, mode="png"):
    if args.api_style == "openai":
        return process_sample_openai_compatible(sample, args)
    elif "gemini-1.5" in args.model_name:
        return process_sample_gemini(sample, args, mode)
    else:
        raise AssertionError()


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


def request_with_fallback(messages, args, routes):
    route_index = 0
    attempts = 0
    last_error = None
    while route_index is not None and route_index < len(routes):
        route = routes[route_index]
        client = build_client(api_key=route.api_key, base_url=route.base_url)
        while attempts < args.max_try:
            try:
                response = client.chat.completions.create(
                    model=route.model_name,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                return response.choices[0].message.content, route, None
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
                if attempts >= args.max_try:
                    next_route_index = route_index + 1 if route_index + 1 < len(routes) else None
                    if next_route_index is None:
                        return f"Failed: {exc}", route, exc
                    route_index = next_route_index
                    break
        else:
            break
    return f"Failed: {last_error}", routes[min(route_index or 0, len(routes) - 1)], last_error


def process_one_sample(sample, args, prompt, routes):
    sample = dict(sample)
    sample.pop("score", None)
    sample.pop("pred", None)
    sample.pop("extracted_res", None)
    sample.pop("error", None)
    sample.pop("failure_stage", None)
    sample.pop("status", None)

    messages = process_sample(sample, args)
    response, used_route, request_error = request_with_fallback(messages, args, routes)
    sample["response"] = response
    sample["used_base_url"] = used_route.base_url
    sample["used_model_name"] = used_route.model_name
    sample["used_route_label"] = used_route.label
    sample["used_route_max_model_len"] = used_route.max_model_len

    if is_failed_response(sample):
        sample["error"] = repr(request_error) if request_error is not None else sample.get("error")
        sample["failure_stage"] = "generation"
        sample["extracted_res"] = "Failed"
        sample["pred"] = "Failed to extract"
        sample["score"] = 0.0
        sample["status"] = "failed_generation"
        return sample

    extractor_model_name = args.extractor_model_name or used_route.model_name
    extractor_client = build_client(
        api_key=args.extractor_api_key if args.extractor_model_name else used_route.api_key,
        base_url=args.extractor_base_url if args.extractor_model_name else used_route.base_url,
    )
    extracted_res = extract_answer(
        sample["question"],
        response,
        prompt,
        model_name=extractor_model_name,
        client=extractor_client,
    )
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
    return sample


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./data/samples.json")
    parser.add_argument("--document_path", type=str, default="./data/documents")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--api_style", type=str, default="openai", choices=["openai", "gemini"])
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--extractor_model_name", type=str, default=None)
    parser.add_argument("--extractor_base_url", type=str, default=None)
    parser.add_argument("--extractor_api_key", type=str, default=None)
    parser.add_argument("--max_pages", type=int, default=120)
    parser.add_argument("--resolution", type=int, default=144)
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
    args.output_path = args.output_path or f'./results/res_{model_slug}.json'
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./tmp", exist_ok=True)

    if args.api_style == "openai":
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
    elif "gemini-1.5" in args.model_name:
        import google.generativeai as genai
        client = genai.GenerativeModel(args.model_name)
        config = genai.types.GenerationConfig(max_output_tokens=args.max_tokens, temperature=args.temperature)
        routes = None
    else:
        raise AssertionError()

    with open(args.extractor_prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()
    samples = load_samples(args)
    
    # 计算已完成和待完成样本
    completed_count = sum(1 for s in samples if should_skip_sample(s))
    total_count = len(samples)
    print(f"进度: {completed_count}/{total_count} 已完成")
    pending_indices = [idx for idx, sample in enumerate(samples) if not should_skip_sample(sample)]

    if args.api_style != "openai":
        raise AssertionError("Parallel routing is implemented for openai-compatible endpoints only.")

    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
        future_to_index = {
            executor.submit(process_one_sample, samples[idx], args, prompt, routes): idx for idx in pending_indices
        }
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Processing"):
            idx = future_to_index[future]
            sample = future.result()
            samples[idx] = sample
            acc, f1 = eval_acc_and_f1(samples)
            print("--------------------------------------")
            print("Question: {}".format(sample["question"]))
            print("Route: {} | {} | {}".format(sample.get("used_route_label"), sample.get("used_base_url"), sample.get("used_model_name")))
            print("Response: {}".format(sample["response"]))
            print("Gt: {}\tPred: {}\tScore: {}".format(sample["answer"], sample["pred"], sample["score"]))
            print("Avg acc: {}".format(acc))
            print("Avg f1: {}".format(f1))
            with open(args.output_path, 'w', encoding="utf-8") as f:
                json.dump(samples, f, ensure_ascii=False)
    
    show_results(samples, show_path=re.sub(r"\.json$", ".txt", args.output_path))
