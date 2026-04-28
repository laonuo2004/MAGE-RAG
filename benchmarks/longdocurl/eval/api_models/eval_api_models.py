import sys
import pathlib

EVAL = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL))
BENCHMARK_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BENCHMARK_ROOT))
BENCHMARKS_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BENCHMARKS_ROOT))
CODE_WORKSPACE = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(CODE_WORKSPACE))

import argparse
import os

import hydra
from omegaconf import DictConfig, OmegaConf
import json
from tqdm import tqdm
import time
from multiprocessing import Pool
import datetime
from openai import OpenAI

from utils_api import *
from utils.hydra_utils import _value
from utils.utils_score_v3 import *
# from model import Gemini15ProInferencer, GPT4oInferencer, QwenVLMaxInferencer, O1PreviewInferencer, QwenMaxInferencer, Gemini31ProInferencer, GPT54Inferencer, ClaudeSonnet46Inferencer
from .model import Inferencer
from pure_ocr_utils import *
from baselines.wrapper import build_context_builder

import logging
logger = logging.getLogger(__name__)

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
_worker_clients = {}


def build_client(llm_provider):
    if llm_provider == "openrouter":
        return OpenAI(api_key=config["api_model"]["access_key"], base_url=config["api_model"]["base_url"])
    if llm_provider == "local":
        return OpenAI(api_key=config["local_model"]["access_key"], base_url=config["local_model"]["base_url"])
    raise ValueError(f"Unsupported llm_provider: {llm_provider}")


def get_worker_client(llm_provider):
    if llm_provider not in _worker_clients:
        _worker_clients[llm_provider] = build_client(llm_provider)
    return _worker_clients[llm_provider]

def preprocess(input_datapath, output_datapath, image_prefix=None):
    dataset = read_jsonl_file(input_datapath)
    logger.info(f"dataset cnt: {len(dataset)}")

    if os.path.exists(output_datapath):
        output_dataset = read_jsonl_file(output_datapath)
        dataset = delete_generate_dataset(dataset, output_dataset)

    if image_prefix is not None:
        for _ in dataset:
            for i, image_path in enumerate(_["images"]):
                _["images"][i] = os.path.join(image_prefix, "/".join(image_path.split("/")[-2:]))

    logger.info(f"dataset cnt need to do: {len(dataset)}")

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
            logger.error(f"error with {e}, response = {response}")
            max_try -= 1
            response = None

    return response

def delete_generate_dataset(dataset, output_dataset):
    finished_question_id_set = set([sample['question_id'] for sample in output_dataset])
    unfinished_dataset = [sample for sample in dataset if sample['question_id'] not in finished_question_id_set]
    return unfinished_dataset


def build_default_results_file(cfg, benchmark_cfg):
    baseline_name = cfg.baselines.name
    # 我们不根据 extractor_model_name 来命名结果，因为没有太大影响
    model_name = benchmark_cfg.qa_model_name.replace("/", "_").replace(":free", "").replace("-", "_")
    return os.path.join(BENCHMARK_ROOT, f"evaluation_results/api_models/results_{baseline_name}.{model_name}.jsonl")


def eval_per_record(args):
    logger.info("--------------------------------------")
    case, output_datapath, ocr_json_dir, image_prefix, qa_llm_provider, qa_model_name, extractor_llm_provider, extractor_model_name, context_builder_name = args
    qa_client = get_worker_client(qa_llm_provider)
    extractor_client = get_worker_client(extractor_llm_provider)

    # inferencer = eval(qa_model_name2inferencer[qa_model_name])()
    inferencer = eval("Inferencer")()

    question = case["question"]
    context_args = argparse.Namespace(
        ocr_json_dir=ocr_json_dir,
        input_format=input_format,
        ocr_backend=ocr_backend,
        baselines={"name": context_builder_name},
    )
    context_builder = build_context_builder(context_args)
    context_bundle = context_builder.build("longdocurl", case, context_args)
    system_prompt = context_bundle.system_prompt
    result = inferencer.infer(context_bundle.prompt, context_bundle.images, qa_model_name, qa_client)

    if result is None:
        return

    # extract concise answer
    with open(extractor_prompt_path) as f:
        extractor_prompt = f.read()
    prompt = system_prompt + extractor_prompt + "\nQuestion: " + question + "\nAnalysis: " + result
    extractor_result = call_llm(extractor_model_name, prompt, None, extractor_client)
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
    case["input_format"] = context_bundle.metadata.get("input_format", input_format)
    case["ocr_backend"] = context_bundle.metadata.get("ocr_backend", ocr_backend if input_format == "ocr" else None)
    case["ocr_pages_used"] = context_bundle.metadata.get("ocr_pages_used", [])
    case["context_builder"] = context_bundle.metadata.get("context_builder", context_builder_name)

    logger.info("\n\n")
    logger.info("Question: {}".format(case["question"]))
    logger.info("Response: {}".format(case["pred"]))
    
    logger.info("Gt: {}\tPred: {}\tScore_v3: {}".format(case["answer"], case["pred"], case["score_v3"]))

    if result is not None:  # Check if result is not None
        try: # not json serialable
            with open(output_datapath, "a") as output_review_file:
                output_review_file.write(json.dumps(case, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"error: {e}")
    else:
        logger.error("error")


# def evaluate(dataset, output_datapath, qa_model_name="gpt4o", extractor_model_name="gpt4o", process_mode="serial", input_format="e2e", ocr_backend="pymupdf", ocr_json_dir=None, extra_infos=None, qa_llm_provider="openrouter", extractor_llm_provider="openrouter", workers=64, context_builder_name=None):
def evaluate(dataset, output_datapath, process_mode="serial", workers=64, ocr_json_dir=None, image_prefix=None, qa_llm_provider="litellm", qa_model_name="Qwen2.5-VL-7B-Instruct", extractor_llm_provider="litellm", extractor_model_name="Qwen2.5-VL-7B-Instruct", context_builder_name="image"):

    # if os.path.exists(output_datapath):
    #     output_dataset = read_jsonl_file(output_datapath)
    #     dataset = delete_generate_dataset(dataset, output_dataset)

    # logger.info(f"dataset cnt: {len(dataset)}")
    if not len(dataset):
        logger.info("No data to process.")
        return

    args_list = []
    for case in dataset:
        args_list.append((case, output_datapath, ocr_json_dir, image_prefix, qa_llm_provider, qa_model_name, extractor_llm_provider, extractor_model_name, context_builder_name))

    start_time = datetime.datetime.now()
    logger.info(f"job start time: {start_time}")

    if process_mode == "serial":
        for args in args_list:
            eval_per_record(args)
    elif process_mode == "parallel":
        with Pool(processes=workers) as pool:  # You can adjust the number of processes as needed
            list(tqdm(pool.imap(eval_per_record, args_list), total=len(args_list)))
    else:
        logger.error("process mode error!")


def build_arg_parser(default_context_builder=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--qa_file', type=str, default="/root/autodl-tmp/ylz/NeurIPS_2026/code/benchmarks/longdocurl/data/LongDocURL.jsonl")
    parser.add_argument('--process_mode', type=str, default="serial") # serial/parallel
    parser.add_argument('--workers', type=int, default=64)
    parser.add_argument('--input_format', type=str, choices=["e2e", "ocr"], default="e2e")
    parser.add_argument('--context_builder', type=str, choices=["image", "ocr"], default=default_context_builder)
    parser.add_argument('--ocr_backend', type=str, choices=["pymupdf"], default="pymupdf")
    parser.add_argument('--ocr_json_dir', type=str, default="/root/autodl-tmp/ylz/NeurIPS_2026/code/benchmarks/longdocurl/data/pdf_jsons/4000-4999")
    parser.add_argument('--image_prefix', type=str, default="/root/autodl-tmp/ylz/NeurIPS_2026/code/benchmarks/longdocurl/data/pdf_pngs/4000-4999")
    parser.add_argument('--qa_llm_provider', type=str, default="openrouter") # openrouter/local
    parser.add_argument('--qa_model_name', type=str, default="gemma-4-26b-a4b-it") # gemini15_pro/claude35_sonnet/qwen_vl_max/gpt4o
    parser.add_argument('--extractor_model_name', type=str, default="gemma-4-26b-a4b-it") # gemini15_pro/claude35_sonnet/qwen_vl_max/gpt4o
    parser.add_argument('--extractor_llm_provider', type=str, default="openrouter") # openrouter/local
    parser.add_argument('--results_file', type=str, default=None)
    return parser

def run_longdocurl(cfg: DictConfig):
    # if args.context_builder is None:
    #     args.context_builder = "ocr" if args.input_format == "ocr" else "image"
    # args.input_format = "ocr" if args.context_builder == "ocr" else "e2e"
    # input_datapath = args.qa_file
    # output_datapath = args.results_file or build_default_results_file(
    #     args.qa_model_name,
    #     args.input_format,
    #     args.ocr_backend,
    # )
    benchmark_cfg = _value(cfg, 'benchmarks', {})
    output_datapath = benchmark_cfg.results_file or build_default_results_file(cfg, benchmark_cfg)
        
    logger.info(f"Output datapath: {output_datapath}")
    
    # client = build_client(args.qa_llm_provider)

    # load data
    # dataset = preprocess(input_datapath, output_datapath)
    # if image paths are not modified in .jsonl file, add image prefix when executed
    dataset = preprocess(benchmark_cfg.qa_file, benchmark_cfg.output_datapath, image_prefix=benchmark_cfg.image_prefix)

    try_cnt = 2
    while try_cnt:
        try_cnt -= 1
        try:
            evaluate(
                dataset,
                output_datapath,
                process_mode=benchmark_cfg.process_mode,
                workers=benchmark_cfg.workers,
                ocr_json_dir=benchmark_cfg.ocr_json_dir,
                image_prefix=benchmark_cfg.image_prefix,
                qa_llm_provider=benchmark_cfg.qa_llm_provider,
                qa_model_name=benchmark_cfg.qa_model_name,
                extractor_llm_provider=benchmark_cfg.extractor_llm_provider,
                extractor_model_name=benchmark_cfg.extractor_model_name,
                context_builder_name=cfg.baselines.name
            )
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            logger.info("Restarting script...")
            time.sleep(1)

    if not os.path.exists(output_datapath):
        logger.error(f"No results generated at: {output_datapath}")
        sys.exit(1)

    acc, f1, = calculate_acc_and_f1(output_datapath)
    logger.info("--------------------------------------")
    logger.info("Avg acc: {}".format(acc))
    logger.info("Avg f1: {}".format(f1))
    return acc, f1


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    return run_longdocurl(args)


if __name__ == "__main__":
    main()
