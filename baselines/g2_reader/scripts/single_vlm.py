# -*- coding: utf-8 -*-
import os
import re
import json
import time
import argparse
import random
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image

from qasper_utils import amem_qa_visual
from utils.visdom_utils import extract_text_from_pdf, split_text, extract_images_from_pdf, encode_image

# =========================
# Utils
# =========================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def stable_int_from_str(s: str) -> int:
    """Stable int from string (cross-run stable)."""
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def get_sample_rng(global_seed: int, sample_id: str) -> random.Random:
    """Per-sample RNG to avoid nondeterminism from multithreading scheduling."""
    seed_i = stable_int_from_str(f"{global_seed}:{sample_id}")
    return random.Random(seed_i)

def safe_eval_documents_field(doc_field):
    """
    documents
    """
    if isinstance(doc_field, list):
        return doc_field
    if isinstance(doc_field, str):
        try:
            return eval(doc_field)
        except Exception:
            try:
                return json.loads(doc_field)
            except Exception:
                return []
    return []


def to_pil(img):
    """Convert image to PIL if possible."""
    if isinstance(img, Image.Image):
        return img
    try:
        return Image.fromarray(img)
    except Exception:
        return None


def clean_surrogates(obj):
    """recursively clean surrogate characters in strings"""
    if isinstance(obj, str):
        return obj.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    elif isinstance(obj, dict):
        return {k: clean_surrogates(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_surrogates(item) for item in obj]
    else:
        return obj


def save_selected_text(seed_text_dir: str, sample_id: str, payload: dict):
    sample_dir = os.path.join(seed_text_dir, sample_id)
    ensure_dir(sample_dir)
    # clean invalid Unicode surrogate characters
    payload = clean_surrogates(payload)
     
    # 1) chunks.json
    with open(os.path.join(sample_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    # 2) context.txt
    with open(os.path.join(sample_dir, "context.txt"), "w", encoding="utf-8") as f:
        f.write(payload.get("context", ""))


def save_selected_images(seed_pdf_dir: str, sample_id: str, pil_images_in_order):
    sample_dir = os.path.join(seed_pdf_dir, sample_id)
    ensure_dir(sample_dir)
    manifest = {"images": []}
    for i, im in enumerate(pil_images_in_order):
        fn = f"img_{i:03d}.png"
        out_path = os.path.join(sample_dir, fn)
        im.save(out_path, format="PNG")
        manifest["images"].append(fn)
    with open(os.path.join(sample_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def load_selected_text(seed_text_dir: str, sample_id: str) -> dict:
    p = os.path.join(seed_text_dir, sample_id, "chunks.json")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_selected_images(seed_pdf_dir: str, sample_id: str):
    sample_dir = os.path.join(seed_pdf_dir, sample_id)
    mani_path = os.path.join(sample_dir, "manifest.json")
    with open(mani_path, "r", encoding="utf-8") as f:
        mani = json.load(f)
    images = []
    for fn in mani.get("images", []):
        images.append(Image.open(os.path.join(sample_dir, fn)).convert("RGB"))
    return images


# =========================
# Core pipeline
# =========================
def build_pdf_paths(sample: dict, doc_root: str, max_pdfs_per_sample: int):
    domains = {
        "paper": "paper_tab",
        "spiqa": "spiqa",
        "feta": "feta_tab",
        "scgqa": "scigraphvqa",
        "slidevqa": "slidevqa",
        "MMLongBench": "MMLongBench",
    }

    sample_id = sample.get("_id", "unknown")
    sample_domain = sample_id.split("_")[0]
    if sample_domain not in domains:
        raise ValueError(f"Unknown sample domain prefix: {sample_domain} (sample_id={sample_id})")

    pdf_names = safe_eval_documents_field(sample.get("documents", "[]"))[:max_pdfs_per_sample]
    doc_root_mid = os.path.join(doc_root, domains[sample_domain])
    final_path = os.path.join(doc_root_mid, "docs")
    pdf_paths = [os.path.join(final_path, doc) for doc in pdf_names]
    return pdf_names, pdf_paths, sample_domain


def extract_and_select_subset(
    sample: dict,
    doc_root: str,
    max_pdfs_per_sample: int,
    max_chunks: int,
    max_images: int,
    global_seed: int,
    seed_pdf_dir: str,
    seed_text_dir: str,
):
    """
    1) read PDF text + images
    2) split_text -> per-sample rng sample chunks
    3) per-sample rng sample images
    4) save to seedxxx/pdf & seedxxx/text
    5) return context(str), selected_pil_images(list), and payload for recording
    """
     
    sample_id = sample.get("_id", "unknown")
    pdf_names, pdf_paths, sample_domain = build_pdf_paths(sample, doc_root, max_pdfs_per_sample)

    rng = get_sample_rng(global_seed, sample_id)

    # ---- 1. extract text and images ----
    t0 = time.time()
    pages = []
    for pdf_path in pdf_paths:
        new_pages = extract_text_from_pdf(pdf_path)
        pages.extend(new_pages)
    full_context = "\n".join(pages)

    images_raw = []
    for pdf_path in pdf_paths:
        new_images = extract_images_from_pdf(pdf_path)
        images_raw.extend(new_images)

    # convert to PIL and discard exceptions
    pil_images = []
    for im in images_raw:
        pim = to_pil(im)
        if pim is not None:
            pil_images.append(pim.convert("RGB"))

    t1 = time.time()

    # ---- 2. chunk sampling (deterministic)----
    chunks = split_text(full_context)
    if len(chunks) > max_chunks:
        # sampling order is fixed: use the order directly from rng.sample (no sorting)
        selected_chunks = rng.sample(chunks, max_chunks)
    else:
        selected_chunks = chunks[:]

    context = "\n".join(selected_chunks)

    # ---- 3. image sampling (deterministic)----
    if len(pil_images) > max_images:
        selected_pil_images = rng.sample(pil_images, max_images)
    else:
        selected_pil_images = pil_images[:]

    t2 = time.time()

    # ---- 4. save subset to disk (for reuse)----
    text_payload = {
        "sample_id": sample_id,
        "domain": sample_domain,
        "global_seed": global_seed,
        "pdf_names": pdf_names,
        "max_pdfs_per_sample": max_pdfs_per_sample,
        "max_chunks": max_chunks,
        "max_images": max_images,
        "num_pages": len(pages),
        "num_total_chunks": len(chunks),
        "num_selected_chunks": len(selected_chunks),
        "num_total_images": len(pil_images),
        "num_selected_images": len(selected_pil_images),
        "selected_chunks": selected_chunks, 
        "context": context,
        "timing": {
            "pdf_extraction": t1 - t0,
            "sampling": t2 - t1,
        },
    }
    save_selected_text(seed_text_dir, sample_id, text_payload)
    save_selected_images(seed_pdf_dir, sample_id, selected_pil_images)
    text_payload_clean = clean_surrogates(text_payload)
    context_clean = clean_surrogates(context)
    return context_clean, selected_pil_images, text_payload_clean


def load_subset_for_inference(seed_pdf_dir: str, seed_text_dir: str, sample: dict):
    sample_id = sample.get("_id", "unknown")
    text_payload = load_selected_text(seed_text_dir, sample_id)
    context = text_payload.get("context", "")
    selected_pil_images = load_selected_images(seed_pdf_dir, sample_id)
    return context, selected_pil_images, text_payload


def run_model_inference(context: str, selected_pil_images, question: str, model_name: str):
    # encode images (encode_image is used as is)
    encoded_images = [encode_image(im) for im in selected_pil_images]
    result = amem_qa_visual(
        context,
        encoded_images,
        question,
        model=model_name,
        temperature=0,
    )
    return result


def process_single_sample(
    sample: dict,
    args,
    seed_root: str,
    seed_results_dir: str,
    seed_pdf_dir: str,
    seed_text_dir: str,
    error_log_file: str = None,
):
    sample_id = sample.get("_id", "unknown")
    question = sample.get("question", "")

    times = []
    time_breakdown = {}
     
    try:
        print(f"[processing] Sample ID: {sample_id}")

        # ---- A. prepare subset (extract or read)----
        t0 = time.time()
        if args.mode in ("extract", "extract_and_infer"):
            context, selected_pil_images, subset_payload = extract_and_select_subset(
                sample=sample,
                doc_root=args.doc_root,
                max_pdfs_per_sample=args.max_pdfs_per_sample,
                max_chunks=args.max_chunks,
                max_images=args.max_images,
                global_seed=args.seed,
                seed_pdf_dir=seed_pdf_dir,
                seed_text_dir=seed_text_dir,
            )
            t1 = time.time()
            time_breakdown["pdf_extraction"] = subset_payload["timing"]["pdf_extraction"]
            time_breakdown["sampling"] = subset_payload["timing"]["sampling"]
            times.append(t1 - t0) 
        elif args.mode == "infer":
            context, selected_pil_images, subset_payload = load_subset_for_inference(
                seed_pdf_dir=seed_pdf_dir,
                seed_text_dir=seed_text_dir,
                sample=sample,
            )
            t1 = time.time()
            time_breakdown["pdf_extraction"] = 0.0
            time_breakdown["sampling"] = 0.0
            times.append(t1 - t0)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        print(
            f"  ├─ subset prepared: chunks={subset_payload.get('num_selected_chunks', 0)} "
            f"images={subset_payload.get('num_selected_images', 0)} "
            f"(seed={args.seed}, mode={args.mode})"
        )

        if args.mode == "extract":
            sample["subset_saved"] = True
            sample["subset_root"] = seed_root
            sample["model_answer"] = None
            sample["times"] = times
            sample["time_breakdown"] = {
                **time_breakdown,
                "model_inference": 0.0,
                "total": sum(times),
            }
            sample["token_usage"] = None
            return sample

            # ---- B. 推理 ----
        t2 = time.time()
            
        result = run_model_inference(context, selected_pil_images, question, args.model)
        t3 = time.time()

        # usage
        answer = result.get("answer", "")
        usage = result.get("usage", {}) or {}
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

        times.append(t3 - t2)
        time_breakdown["model_inference"] = t3 - t2

        # ---- C. write back sample ----
        sample["subset_saved"] = True
        sample["subset_root"] = seed_root
        sample["subset_meta"] = {
            "seed": args.seed,
            "mode": args.mode,
            "pdf_dir": os.path.relpath(os.path.join(seed_pdf_dir, sample_id), seed_root),
            "text_dir": os.path.relpath(os.path.join(seed_text_dir, sample_id), seed_root),
        }

        sample["times"] = times
        sample["time_breakdown"] = {
            **time_breakdown,
            "total": sum(time_breakdown.values()),
        }
        sample["token_usage"] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "note": "input_tokens includes both text and image tokens (as reported by API)",
        }
        sample["model_answer"] = answer

        print(
            f"  └─ inference completed: input={input_tokens}, output={output_tokens}, total={total_tokens}, "
            f"infer_time={time_breakdown['model_inference']:.2f}s"
        )
        return sample

    except Exception as e:
        error_msg = f"[error] Sample ID: {sample_id} - {type(e).__name__}: {e}"
        print(error_msg)
        
        # save error log to file
        if error_log_file:
            error_record = {
                "sample_id": sample_id,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "sample": sample
            }
            try:
                with open(error_log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(error_record, ensure_ascii=False) + "\n")
            except:
                pass
        
        return None


# =========================
# Main
# =========================
def load_data(path: str):
    print(f"[INFO] start loading data: {path}")
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    print(f"[INFO] data loaded, total {len(data)} samples")
    return data


def dataset_name_from_input(input_path: str) -> str:
    bn = os.path.basename(input_path)
    bn = re.sub(r"\.(jsonl|json)$", "", bn)
    return bn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--doc_root", type=str, required=True)
    parser.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL", "qwen3-vl-32b-instruct"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, choices=["extract", "infer", "extract_and_infer"], default="extract_and_infer")
    parser.add_argument("--max_pdfs_per_sample", type=int, default=5)
    parser.add_argument("--max_chunks", type=int, default=10)
    parser.add_argument("--max_images", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--out_root", type=str, default="Single-llm/results")
    args = parser.parse_args()

    seed_root = os.path.join(args.out_root, f"seed{args.seed:03d}")


    dataset_name = dataset_name_from_input(args.input_path)
    seed_results_dir = os.path.join(seed_root, "results", dataset_name, args.model)
    seed_pdf_dir = os.path.join(seed_root, "pdf")
    seed_text_dir = os.path.join(seed_root, "text")

    ensure_dir(seed_results_dir)
    ensure_dir(seed_pdf_dir)
    ensure_dir(seed_text_dir)

    output_path = os.path.join(seed_results_dir, "results.jsonl")
    stats_file = os.path.join(seed_results_dir, "stats.json")
    error_log_file = os.path.join(seed_results_dir, "errors.jsonl")

    print("=" * 80)
    print("[INFO] running configuration")
    print(f"  - input_path: {args.input_path}")
    print(f"  - doc_root:   {args.doc_root}")
    print(f"  - model:      {args.model}")
    print(f"  - seed:       {args.seed}")
    print(f"  - mode:       {args.mode}")
    print(f"  - out_root:   {args.out_root}")
    print(f"  - seed_root:  {seed_root}")
    print(f"  - results:    {output_path}")
    print(f"  - error_log:  {error_log_file}")
    print("=" * 80)
      
    data = load_data(args.input_path)

    # clear output file
    with open(output_path, "w", encoding="utf-8") as f:
        pass

    start_time = time.time()

    completed_count = 0
    failed_count = 0

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0

    total_extraction_time = 0.0
    total_sampling_time = 0.0
    total_inference_time = 0.0

    futures = []
    # serial debugging
    if args.num_workers == 1:
        for sample in data:
              
            result = process_single_sample(
                sample,
                args,
                seed_root,
                seed_results_dir,
                seed_pdf_dir,
                seed_text_dir,
                error_log_file,
            )
            if result is None:
                failed_count += 1
                continue

            completed_count += 1
            with open(output_path, "a", encoding="utf-8") as f_out:
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            for sample in data:
                futures.append(
                    executor.submit(
                        process_single_sample,
                        sample,
                        args,
                        seed_root,
                        seed_results_dir,
                        seed_pdf_dir,
                        seed_text_dir,
                        error_log_file,
                    )
                )

            with open(output_path, "a", encoding="utf-8") as f_out:
                for future in tqdm(as_completed(futures), total=len(futures), desc="processing progress"):
                    result = future.result()
                    if result is None:
                        failed_count += 1
                        continue

                    completed_count += 1
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

                    # statistics
                    tb = result.get("time_breakdown") or {}
                    total_extraction_time += float(tb.get("pdf_extraction", 0.0))
                    total_sampling_time += float(tb.get("sampling", 0.0))
                    total_inference_time += float(tb.get("model_inference", 0.0))

                    tu = result.get("token_usage") or {}
                    total_input_tokens += int(tu.get("input_tokens", 0) or 0)
                    total_output_tokens += int(tu.get("output_tokens", 0) or 0)
                    total_tokens += int(tu.get("total_tokens", 0) or 0)

    end_time = time.time()
    total_time = end_time - start_time

    # save stats
    stats = {
        "dataset": dataset_name,
        "model": args.model,
        "seed": args.seed,
        "mode": args.mode,
        "total_samples": len(data),
        "completed": completed_count,
        "failed": failed_count,
        "token_usage": {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "avg_input_tokens": (total_input_tokens / completed_count) if completed_count > 0 else 0,
            "avg_output_tokens": (total_output_tokens / completed_count) if completed_count > 0 else 0,
            "avg_total_tokens": (total_tokens / completed_count) if completed_count > 0 else 0,
        },
        "time_usage": {
            "total_wall_time": total_time,
            "total_extraction_time": total_extraction_time,
            "total_sampling_time": total_sampling_time,
            "total_inference_time": total_inference_time,
            "avg_wall_time_per_sample": (total_time / completed_count) if completed_count > 0 else 0,
            "avg_inference_time_per_sample": (total_inference_time / completed_count) if completed_count > 0 else 0,
        },
        "paths": {
            "seed_root": seed_root,
            "results_jsonl": output_path,
            "pdf_dir": seed_pdf_dir,
            "text_dir": seed_text_dir,
            "stats_json": stats_file,
            "error_log": error_log_file,
        },
    }

    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("[completed] all samples processed!")
    print(f"  - total samples: {len(data)}")
    print(f"  - successful: {completed_count}")
    print(f"  - failed: {failed_count}")
    print(f"  - output file: {output_path}")
    print(f"  - statistics file: {stats_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
