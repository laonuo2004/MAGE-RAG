import os
import re
import json
import base64
import argparse
import fitz
from io import BytesIO

from PIL import Image
from tqdm import tqdm

from eval.extract_answer import build_client, extract_answer
from eval.eval_score import eval_score, eval_acc_and_f1, show_results


cached_image_list = dict()


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
    if os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            return json.load(f)
    with open(args.input_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if args.limit is not None:
        samples = samples[: args.limit]
    return samples


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./data/samples.json")
    parser.add_argument("--document_path", type=str, default="./data/documents")
    parser.add_argument("--model_name", type=str, required=True)
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
    args = parser.parse_args()

    model_slug = re.sub(r"[^0-9a-zA-Z._-]+", "_", args.model_name)
    args.output_path = args.output_path or f'./results/res_{model_slug}.json'
    args.extractor_model_name = args.extractor_model_name or args.model_name
    args.extractor_base_url = args.extractor_base_url or args.base_url
    args.extractor_api_key = args.extractor_api_key or args.api_key
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./tmp", exist_ok=True)

    if args.api_style == "openai":
        client = build_client(api_key=args.api_key, base_url=args.base_url)
        extractor_client = build_client(
            api_key=args.extractor_api_key,
            base_url=args.extractor_base_url,
        )
    elif "gemini-1.5" in args.model_name:
        import google.generativeai as genai
        client = genai.GenerativeModel(args.model_name)
        config = genai.types.GenerationConfig(max_output_tokens=args.max_tokens, temperature=args.temperature)
        extractor_client = build_client(
            api_key=args.extractor_api_key,
            base_url=args.extractor_base_url,
        )
    else:
        raise AssertionError()

    with open(args.extractor_prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()
    samples = load_samples(args)
    
    # 计算已完成和待完成样本
    completed_count = sum(1 for s in samples if "score" in s)
    total_count = len(samples)
    print(f"进度: {completed_count}/{total_count} 已完成")

    for idx, sample in enumerate(tqdm(samples, desc="Processing")):
        if "score" in sample:
            score = sample["score"]
        else:
            messages = process_sample(sample, args)
            
            try_cnt = 0
            is_success = False
            while True:
                try:
                    if args.api_style == "openai":
                        response = client.chat.completions.create(
                            model=args.model_name,
                            messages=messages,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature
                        )
                        response = response.choices[0].message.content
                    elif "gemini-1.5" in args.model_name:
                        try:
                            response = client.generate_content(messages, generation_config=config)
                        except:
                            print("Payload oversize! Use File API instead.")
                            messages = process_sample(sample, args, mode="file")
                            response = client.generate_content(messages, generation_config=config)
                        response.resolve()
                        response = response.text.strip()
                    else:
                        pass
                    is_success = True
                except:
                    try_cnt += 1
                    response = "Failed"
                if is_success or try_cnt>args.max_try:
                    break
                
            sample["response"] = response
            extracted_res = extract_answer(
                sample["question"],
                response,
                prompt,
                model_name=args.extractor_model_name,
                client=extractor_client,
            )
            sample["extracted_res"] = extracted_res
            # try:
            print(extracted_res)
            pred_ans = extracted_res.split("Answer format:")[0].split("Extracted answer:")[1].strip()
            score = eval_score(sample["answer"], pred_ans, sample["answer_format"])
            # except:
            #     pred_ans = "Failed to extract"
            #     score = 0.0
            sample["pred"] = pred_ans
            sample["score"] = score

        acc, f1 = eval_acc_and_f1(samples)
        print("--------------------------------------")
        print("Question: {}".format(sample["question"]))
        print("Response: {}".format(sample["response"]))
        print("Gt: {}\tPred: {}\tScore: {}".format(sample["answer"], sample["pred"], sample["score"]))
        print("Avg acc: {}".format(acc))
        print("Avg f1: {}".format(f1))
        
        # 每处理完一个样本就保存，确保断电不丢失进度
        with open(args.output_path, 'w', encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False)
    
    show_results(samples, show_path=re.sub(r"\.json$", ".txt", args.output_path))
