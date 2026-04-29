import sys
import pathlib

from importlib_metadata import metadata

EVAL = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL))
BENCHMARK_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BENCHMARK_ROOT))
BENCHMARKS_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BENCHMARKS_ROOT))
CODE_WORKSPACE = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(CODE_WORKSPACE))

import os

from omegaconf import DictConfig, OmegaConf

BENCHMARK_ROOT = pathlib.Path(__file__).resolve().parents[2]
import json
from tqdm import tqdm
import time
from multiprocessing import Pool
import datetime
from openai import OpenAI

from utils_api import *
from utils.utils_score_v3 import *
from utils.calculate_metrics import calculate_metrics
from utils.calculate_metrics_fine_grained import calculate_metrics_fine_grained
# from model import Gemini15ProInferencer, GPT4oInferencer, QwenVLMaxInferencer, O1PreviewInferencer, QwenMaxInferencer, Gemini31ProInferencer, GPT54Inferencer, ClaudeSonnet46Inferencer
from pure_ocr_utils import *
from baselines.wrapper import build_context_builder
from utils.logging_utils import apply_logging_config

import logging
logger = logging.getLogger("longdocurl.eval_api_models")

vision_system_prompt = "You are an expert in visual document question-answering, please answer our questions based on the given images.\n"
text_system_prompt = "You are an expert in document question-answering, please answer our questions based on the extracted text from the given pages.\n"

# TODO
project_prefix = "/root/autodl-tmp/ylz/NeurIPS_2026/code/benchmarks/longdocurl/"

extractor_prompt_path = os.path.join(project_prefix, "eval/prompt_for_answer_extraction.md")
score_sample_file = BENCHMARK_ROOT / "evaluation_results/scores_sample_fine_grained.json"

# model_name2inferencer = {"gpt4o": "GPT4oInferencer", "gemini15_pro": "Gemini15ProInferencer", "qwen_vl_max": "QwenVLMaxInferencer", \
#     "o1_preview": "O1PreviewInferencer", "qwen_max": "QwenMaxInferencer", "gemini-3.1-pro-preview": "Gemini31ProInferencer", \
#     "gpt-5.4": "GPT54Inferencer", "claude-sonnet-4-6": "ClaudeSonnet46Inferencer", "google/gemma-3-27b-it:free": "OpenRouterInferencer", \
#     "google/gemma-4-26b-a4b-it:free": "OpenRouterInferencer", "google/gemma-4-26b-a4b-it": "OpenRouterInferencer"}

prompt_sign = True
client = None
_worker_clients = {}

def preprocess(input_datapath, output_datapath, image_prefix=None):
    dataset = read_jsonl_file(input_datapath)
    logger.info(f"Dataset Count: {len(dataset)}")

    if os.path.exists(output_datapath):
        output_dataset = read_jsonl_file(output_datapath)
        dataset = delete_generate_dataset(dataset, output_dataset)

    if image_prefix is not None:
        for _ in dataset:
            for i, image_path in enumerate(_["images"]):
                _["images"][i] = os.path.join(image_prefix, "/".join(image_path.split("/")[-2:]))

    logger.info(f"Dataset Count Need To Do: {len(dataset)}")

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
    return call_llm_messages(model_name, msgs, client_obj, temperature=temperature, seed=seed, max_tokens=max_tokens)


def call_llm_messages(model_name, messages, client_obj, temperature=0.1, seed=42, max_tokens=4096):
    response = None
    max_try = 3
    while response is None and max_try > 0:
        try:
            # TODO
            completion = client_obj.chat.completions.create(model=model_name, messages=messages, temperature=0., max_completion_tokens=max_tokens)
            logger.debug(f"Raw LLM response: {completion.choices[0].message}")
            response = completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error With {e}, Response = {response}")
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
    return os.path.join(BENCHMARK_ROOT, f"evaluation_results/api_models/results_{baseline_name}_{model_name}.jsonl")


def eval_per_record(task):
    cfg, case, output_datapath = task
    benchmark_cfg = cfg.get('benchmarks', {})
    # 统一使用 litellm 路由
    client = OpenAI(api_key=cfg.litellm.api_key, base_url=cfg.litellm.base_url)
    qa_model_name = benchmark_cfg.get('qa_model_name', None)
    extractor_model_name = benchmark_cfg.get('extractor_model_name', None)

    question = case["question"]
    # ========== 这一部分抽象程度较高，需要仔细分析理解 ==========
    context_builder = build_context_builder(cfg)
    messages = context_builder.build("longdocurl", case)
    # =========================================================
    result = call_llm_messages(qa_model_name, messages, client)
    logger.debug(f"LLM response for question_id {case.get('question_id', 'unknown')}: {result}")
    if result is None:
        logger.warning(f"LLM returned None for question_id {case.get('question_id', 'unknown')}. Skipping this case.")
        return

    # extract concise answer
    with open(extractor_prompt_path) as f:
        extractor_prompt = f.read()
    prompt = extractor_prompt + "\nQuestion: " + question + "\nAnalysis: " + result
    
    logger.debug(f"Extractor prompt for question_id {case.get('question_id', 'unknown')}: {prompt}")
    extractor_result = call_llm(extractor_model_name, prompt, None, client)
    logger.debug(f"Extractor LLM response for question_id {case.get('question_id', 'unknown')}: {extractor_result}")
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
        logger.warning(f"Failed to extract concise answer for question_id {case.get('question_id', 'unknown')}.")
        score_v3 = 0.0
    else:
        score_v3 = eval_score(case["answer"], pred_ans, case["answer_format"])
        
    case["detailed_response"] = result
    case["pred"] = pred_ans
    case["score_v3"] = score_v3
    # case["input_format"] = metadata.get("input_format")
    # case["ocr_backend"] = metadata.get("ocr_backend")
    # case["ocr_pages_used"] = metadata.get("ocr_pages_used", [])
    # case["context_builder"] = metadata.get("context_builder")

    # logger.info("\n\n")
    # logger.info("Question: {}".format(case["question"]))
    # logger.info("Response: {}".format(case["pred"]))
    
    # logger.info("GT: {}\tPred: {}\tScore V3: {}".format(case["answer"], case["pred"], case["score_v3"]))
    # 为了避免多线程时日志输出混乱，我们将其改为 debug 级别
    logger.debug("Question: {}".format(case["question"]))
    logger.debug("Response: {}".format(case["pred"]))
    
    logger.debug("GT: {}\tPred: {}\tScore V3: {}".format(case["answer"], case["pred"], case["score_v3"]))

    if result is not None:  # Check if result is not None
        try: # not json serialable
            with open(output_datapath, "a") as output_review_file:
                output_review_file.write(json.dumps(case, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Error: {e}")
    else:
        logger.error("Error")

def evaluate(cfg, dataset, output_datapath):
    benchmark_cfg = cfg.get('benchmarks', {})
    process_mode = benchmark_cfg.get('process_mode', 'serial')
    workers = benchmark_cfg.get('workers', 64)

    logger.info(f"Evaluation Process Mode: {process_mode}")
    if process_mode == "parallel":
        logger.info(f"Number Of Worker Processes: {workers}")
    
    if not len(dataset):
        logger.info("No Data To Process.")
        return

    tasks = []
    for case in dataset:
        tasks.append((cfg, case, output_datapath))

    baseline_name = cfg.baselines.name
    logger.info(f"Using Baseline: {baseline_name}")

    start_time = datetime.datetime.now()
    logger.info(f"Job Start Time: {start_time}")

    if process_mode == "serial":
        for task in tasks:
            eval_per_record(task)
    elif process_mode == "parallel":
        # Disable DEBUG logs for console only in parallel mode to protect tqdm
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler) and handler.level in (logging.NOTSET, logging.DEBUG):
                handler.setLevel(logging.INFO)

        with Pool(processes=workers) as pool:  # You can adjust the number of processes as needed
            # Use imap_unordered so tqdm is updated as worker tasks finish
            for _ in tqdm(pool.imap_unordered(eval_per_record, tasks), total=len(tasks), mininterval=0.5):
                pass

    else:
        logger.error("Process Mode Error!")


def run_longdocurl(cfg: DictConfig):
    apply_logging_config(cfg)

    benchmark_cfg = cfg.benchmarks
    output_datapath = benchmark_cfg.get("results_file") or build_default_results_file(cfg, benchmark_cfg)
    os.makedirs(os.path.dirname(output_datapath), exist_ok=True)
        
    logger.info(f"Output Datapath: {output_datapath}")

    dataset = preprocess(benchmark_cfg.qa_file, output_datapath, benchmark_cfg.image_prefix)

    try:
        evaluate(cfg, dataset, output_datapath)
    except Exception as e:
        logger.error(f"An Error Occurred During Evaluation: {e}")
        logger.info("Evaluation Failed.")
        return

    if not os.path.exists(output_datapath):
        logger.error(f"No Results Generated At: {output_datapath}")
        sys.exit(1)

    metrics = calculate_metrics(output_datapath)
    fine_grained_metrics = calculate_metrics_fine_grained(output_datapath, score_sample_file)
    logger.info(f"Metrics: {metrics}")
    logger.info(f"Fine-grained metrics saved under: {pathlib.Path(output_datapath).with_suffix('')}")

    return {
        "results_file": output_datapath,
        "metrics": metrics,
        "fine_grained_metrics": fine_grained_metrics,
    }
