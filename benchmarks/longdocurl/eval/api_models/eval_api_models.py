# TODO:
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).absolute().parent.parent.parent))

import argparse
import os

import json
from tqdm import tqdm
import time
from multiprocessing import Pool
import datetime
from openai import OpenAI

from eval.utils_api import *
from utils.utils_score_v3 import *
# from model import Gemini15ProInferencer, GPT4oInferencer, QwenVLMaxInferencer, O1PreviewInferencer, QwenMaxInferencer, Gemini31ProInferencer, GPT54Inferencer, ClaudeSonnet46Inferencer
from model import Inferencer
from pure_ocr_utils import *

vision_system_prompt = "You are an expert in visual document question-answering, please answer our questions based on the given images.\n"
text_system_prompt = "You are an expert in document question-answering, please answer our questions based on the extracted text from the given pages.\n"

# TODO
project_prefix = "/root/autodl-tmp/ylz/NeurIPS_2026/code/benchmarks/longdocurl/"

config_file = os.path.join(project_prefix, "config/api_config.json")
extractor_prompt_path = os.path.join(project_prefix, "eval/prompt_for_answer_extraction.md")

with open(config_file, "r", encoding="utf-8") as rf:
    config = json.load(rf)

# model_name2inferencer = {"gpt4o": "GPT4oInferencer", "gemini15_pro": "Gemini15ProInferencer", "qwen_vl_max": "QwenVLMaxInferencer", \
#     "o1_preview": "O1PreviewInferencer", "qwen_max": "QwenMaxInferencer", "gemini-3.1-pro-preview": "Gemini31ProInferencer", \
#     "gpt-5.4": "GPT54Inferencer", "claude-sonnet-4-6": "ClaudeSonnet46Inferencer", "google/gemma-3-27b-it:free": "OpenRouterInferencer", \
#     "google/gemma-4-26b-a4b-it:free": "OpenRouterInferencer", "google/gemma-4-26b-a4b-it": "OpenRouterInferencer"}

prompt_sign = True
client = None
_worker_client = None


def build_client(llm_provider):
    if llm_provider == "openrouter":
        return OpenAI(api_key=config["api_model"]["access_key"], base_url=config["api_model"]["base_url"])
    if llm_provider == "local":
        return OpenAI(api_key=config["local_model"]["access_key"], base_url=config["local_model"]["base_url"])
    raise ValueError(f"Unsupported llm_provider: {llm_provider}")


def get_worker_client(llm_provider):
    global _worker_client
    if _worker_client is None:
        _worker_client = build_client(llm_provider)
    return _worker_client

def preprocess(input_datapath, output_datapath, image_prefix=None):
    dataset = read_jsonl_file(input_datapath)
    print("dataset cnt: ", len(dataset))

    if os.path.exists(output_datapath):
        output_dataset = read_jsonl_file(output_datapath)
        dataset = delete_generate_dataset(dataset, output_dataset)

    if image_prefix is not None:
        for _ in dataset:
            for i, image_path in enumerate(_["images"]):
                _["images"][i] = os.path.join(image_prefix, "/".join(image_path.split("/")[-2:]))

    print("dataset cnt need to do: ", len(dataset))

    return dataset

def read_jsonl_file(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as jsonl_file:
        for i, line in enumerate(jsonl_file):
            data_dict = json.loads(line.strip())
            if 'question_id' not in data_dict:
                data_dict['question_id'] = i
            data.append(data_dict)
    return data

def call_llm(model_name, prompt, urls, client_obj, temperature=0.1, seed=42, max_tokens=4096):
    msgs = get_msg_format(prompt, urls)
    response = None
    max_try = 2
    while response is None and max_try > 0:
        try:
            # TODO
            # completion = client.chat.completions.create(model="gpt-4o-0513", messages=msgs, temperature=0.)
            completion = client_obj.chat.completions.create(model=model_name, messages=msgs, temperature=0.)
            response = completion.choices[0].message.content
        except Exception as e:
            print(f"error with {e}, response = {response}")
            max_try -= 1
            response = None

    return response

def delete_generate_dataset(dataset, output_dataset):
    finished_question_id_set = set([sample['question_id'] for sample in output_dataset])
    unfinished_dataset = [sample for sample in dataset if sample['question_id'] not in finished_question_id_set]
    return unfinished_dataset


def build_default_results_file(model_name, input_format, ocr_backend):
    model_key = model_name.replace("/", "_").replace(":free", "").replace("-", "_")
    suffix = input_format if input_format == "e2e" else f"{input_format}_{ocr_backend}"
    return os.path.join(project_prefix, f"evaluation_results/api_models/results_{model_key}_{suffix}.jsonl")


def eval_per_record(args):
    print("--------------------------------------")
    case, output_datapath, model_name, input_format, ocr_backend, ocr_json_dir, llm_provider = args
    record_client = get_worker_client(llm_provider)

    # inferencer = eval(model_name2inferencer[model_name])()
    inferencer = eval("Inferencer")()

    question = case["question"]
    ocr_pages_used = []

    if input_format == "e2e":
        system_prompt = vision_system_prompt
        prompt = system_prompt + "Following is our question: \n" + f"<question>{question}</question>" + "\n"
        result = inferencer.infer(prompt, case["images"], model_name, record_client)
    elif input_format == "ocr" and ocr_backend == "pymupdf":
        ocr_prompt, ocr_pages_used = get_pure_ocr_prompt_pymupdf(
            case["doc_no"],
            images=case.get("images"),
            ocr_json_dir=ocr_json_dir,
            start_page=case["start_end_idx"][0],
            end_page=case["start_end_idx"][1],
        )
        system_prompt = text_system_prompt
        prompt = (
            system_prompt
            + "Following is our question: \n"
            + f"<question>{question}</question>\n"            
            + "Following are the extracted texts from the selected document pages:\n"
            + ocr_prompt
        )
        result = inferencer.infer(prompt, None, model_name, record_client)
    else:
        raise ValueError(f"Unsupported input format/backend combination: {input_format}/{ocr_backend}")

    if result is None:
        return

    # extract concise answer
    with open(extractor_prompt_path) as f:
        extractor_prompt = f.read()
    prompt = system_prompt + extractor_prompt + "\nQuestion: " + question + "\nAnalysis: " + result
    extractor_result = call_llm(model_name, prompt, None, record_client)
    try:
        import re
        concise_answer = re.findall(r"<concise_answer>(.*?)</concise_answer>", extractor_result, re.DOTALL)[0]
        answer_format = re.findall(r"<answer_format>(.*?)</answer_format>", extractor_result, re.DOTALL)[0]
    except:
        concise_answer = "Fail to extract"
        answer_format = "None"

    # calculate scores
    try:
        # pred_ans = eval(concise_answer)
        pred_ans = eval(concise_answer) if not isinstance(eval(concise_answer), set) else list(eval(concise_answer))
    except:
        pred_ans = concise_answer
    if pred_ans == "Fail to extract":
        score_v3 = 0.0
    else:
        score_v3 = eval_score(case["answer"], pred_ans, case["answer_format"])
        
    case["detailed_response"] = result
    case["pred"] = pred_ans
    case["score_v3"] = score_v3
    case["input_format"] = input_format
    case["ocr_backend"] = ocr_backend if input_format == "ocr" else None
    case["ocr_pages_used"] = ocr_pages_used

    print("\n\n")
    print("Question: {}".format(case["question"]))
    print("Response: {}".format(case["pred"]))
    
    print("Gt: {}\tPred: {}\tScore_v3: {}".format(case["answer"], case["pred"], case["score_v3"]))

    if result is not None:  # Check if result is not None
        try: # not json serialable
            with open(output_datapath, "a") as output_review_file:
                output_review_file.write(json.dumps(case, ensure_ascii=False) + "\n")
        except Exception as e:
            print("error: ", e)
    else:
        print("error")


def evaluate(dataset, output_datapath, model_name="gpt4o", process_mode="serial", input_format="e2e", ocr_backend="pymupdf", ocr_json_dir=None, extra_infos=None, llm_provider="openrouter"):

    if os.path.exists(output_datapath):
        output_dataset = read_jsonl_file(output_datapath)
        dataset = delete_generate_dataset(dataset, output_dataset)

    print("dataset cnt: ", len(dataset))
    if not len(dataset):
        return

    args_list = []
    for case in dataset:
        args_list.append((case, output_datapath, model_name, input_format, ocr_backend, ocr_json_dir, llm_provider))

    start_time = datetime.datetime.now()
    print("job start time:", start_time)

    if process_mode == "serial":
        for args in args_list:
            eval_per_record(args)
    elif process_mode == "parallel":
        with Pool(processes=8) as pool:  # You can adjust the number of processes as needed
            list(tqdm(pool.imap(eval_per_record, args_list), total=len(args_list)))
    else:
        print("process mode error!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--qa_file', type=str, default="/root/autodl-tmp/ylz/NeurIPS_2026/code/benchmarks/longdocurl/data/LongDocURL.jsonl")
    parser.add_argument('--process_mode', type=str, default="serial") # serial/parallel
    parser.add_argument('--llm_provider', type=str, default="openrouter") # openrouter/local
    parser.add_argument('--input_format', type=str, choices=["e2e", "ocr"], default="e2e")
    parser.add_argument('--ocr_backend', type=str, choices=["pymupdf"], default="pymupdf")
    parser.add_argument('--ocr_json_dir', type=str, default="/root/autodl-tmp/ylz/NeurIPS_2026/code/benchmarks/longdocurl/data/pdf_jsons/4000-4999")
    parser.add_argument('--image_prefix', type=str, default="/root/autodl-tmp/ylz/NeurIPS_2026/code/benchmarks/longdocurl/data/pdf_pngs/4000-4999")
    parser.add_argument('--model_name', type=str, default="gemma-4-26b-a4b-it") # gemini15_pro/claude35_sonnet/qwen_vl_max/gpt4o
    parser.add_argument('--results_file', type=str, default=f"/root/autodl-tmp/ylz/NeurIPS_2026/code/benchmarks/longdocurl/evaluation_results/api_models/results_{parser.parse_args().model_name.replace('/', '_').replace(':free', '').replace('-', '_')}_{parser.parse_args().input_format}.jsonl")

    args = parser.parse_args()

    input_datapath = args.qa_file
    output_datapath = args.results_file
    
    client = build_client(args.llm_provider)

    # load data
    # dataset = preprocess(input_datapath, output_datapath)
    # if image paths are not modified in .jsonl file, add image prefix when executed
    dataset = preprocess(input_datapath, output_datapath, image_prefix=args.image_prefix)

    try_cnt = 2
    while try_cnt:
        try_cnt -= 1
        try:
            evaluate(
                dataset,
                output_datapath,
                model_name=args.model_name,
                process_mode=args.process_mode,
                input_format=args.input_format,
                ocr_backend=args.ocr_backend,
                ocr_json_dir=args.ocr_json_dir,
                llm_provider=args.llm_provider,
            )
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Restarting script...")
            time.sleep(1)

    if not os.path.exists(output_datapath):
        print(f"No results generated at: {output_datapath}")
        sys.exit(1)

    acc, f1, = calculate_acc_and_f1(output_datapath)
    print("--------------------------------------")
    print("Avg acc: {}".format(acc))
    print("Avg f1: {}".format(f1))
