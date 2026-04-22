import argparse
import json
import os
import re

import fitz
from tqdm import tqdm

from env_utils import get_config_value, load_local_env
from eval.extract_answer import build_client, extract_answer
from eval.eval_score import eval_acc_and_f1, eval_score, show_results


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
    if os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            return json.load(f)
    with open(args.input_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if args.limit is not None:
        samples = samples[: args.limit]
    return samples


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

    if not args.model_name:
        raise ValueError("Missing model name. Set --model_name or MODEL_NAME in .env.mmlongbench/.env.")

    model_slug = re.sub(r"[^0-9a-zA-Z._-]+", "_", args.model_name)
    args.output_path = args.output_path or f"./results/res_text_{model_slug}.json"
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./tmp", exist_ok=True)

    client = build_client(api_key=args.api_key, base_url=args.base_url)
    extractor_client = build_client(
        api_key=args.extractor_api_key,
        base_url=args.extractor_base_url,
    )

    with open(args.extractor_prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()

    samples = load_samples(args)
    completed_count = sum(1 for s in samples if "score" in s)
    total_count = len(samples)
    print(f"Progress: {completed_count}/{total_count} completed")

    for sample in tqdm(samples, desc="Processing OCR/Text route"):
        if "score" in sample:
            acc, f1 = eval_acc_and_f1(samples)
            print("--------------------------------------")
            print("Question: {}".format(sample.get("question")))
            print("Response: {}".format(sample.get("response", "")))
            print("Gt: {}\tPred: {}\tScore: {}".format(sample.get("answer"), sample.get("pred"), sample.get("score")))
            print("Avg acc: {}".format(acc))
            print("Avg f1: {}".format(f1))
            continue

        messages = build_text_prompt(sample, args)

        try_cnt = 0
        response = "Failed"
        while try_cnt <= args.max_try:
            try:
                completion = client.chat.completions.create(
                    model=args.model_name,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                response = completion.choices[0].message.content
                break
            except Exception as exc:
                try_cnt += 1
                response = f"Failed: {exc}"

        sample["response"] = response
        extracted_res = extract_answer(
            sample["question"],
            response,
            prompt,
            model_name=args.extractor_model_name,
            client=extractor_client,
        )
        sample["extracted_res"] = extracted_res
        try:
            pred_ans = extracted_res.split("Answer format:")[0].split("Extracted answer:")[1].strip()
            score = eval_score(sample["answer"], pred_ans, sample["answer_format"])
        except Exception:
            pred_ans = "Failed to extract"
            score = 0.0
        sample["pred"] = pred_ans
        sample["score"] = score

        acc, f1 = eval_acc_and_f1(samples)
        print("--------------------------------------")
        print("Question: {}".format(sample["question"]))
        print("Response: {}".format(sample["response"]))
        print("Gt: {}\tPred: {}\tScore: {}".format(sample["answer"], sample["pred"], sample["score"]))
        print("Avg acc: {}".format(acc))
        print("Avg f1: {}".format(f1))

        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False)

    show_results(samples, show_path=re.sub(r"\.json$", ".txt", args.output_path))
