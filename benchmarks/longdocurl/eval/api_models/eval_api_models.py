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

import logging
logger = logging.getLogger(__name__)

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


def build_client(llm_provider, llm_provider_cfg=None):
    llm_provider_cfg = llm_provider_cfg or {}
    selected_cfg = llm_provider_cfg.get(llm_provider, {})
    base_url = selected_cfg.get("base_url", None)
    api_key = selected_cfg.get("api_key", None)
    if base_url is None or api_key is None:
        raise ValueError(f"Base URL or API key not found for llm_provider: {llm_provider}. Set llm_providers.{llm_provider}.")
    return OpenAI(api_key=api_key, base_url=base_url)


def get_worker_client(llm_provider, llm_provider_cfg=None):
    if llm_provider not in _worker_clients:
        _worker_clients[llm_provider] = build_client(llm_provider, llm_provider_cfg)
    return _worker_clients[llm_provider]

def get_model_name(model_name, llm_provider, llm_provider_cfg=None):
    selected_cfg = llm_provider_cfg.get(llm_provider, {})
    model_mapping = selected_cfg.get("model_mapping", {})
    return model_mapping.get(model_name, model_name)

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
    max_try = 2
    while response is None and max_try > 0:
        try:
            # TODO
            completion = client_obj.chat.completions.create(model=model_name, messages=messages, temperature=0., max_completion_tokens=max_tokens)
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
    llm_provider_cfg = cfg.get('llm_providers', {})
    qa_client = get_worker_client(benchmark_cfg.get('qa_llm_provider'), llm_provider_cfg)
    qa_model_name = get_model_name(benchmark_cfg.get('qa_model_name'), benchmark_cfg.get('qa_llm_provider'), llm_provider_cfg)
    extractor_client = get_worker_client(benchmark_cfg.get('extractor_llm_provider'), llm_provider_cfg)
    extractor_model_name = get_model_name(benchmark_cfg.get('extractor_model_name'), benchmark_cfg.get('extractor_llm_provider'), llm_provider_cfg)

    question = case["question"]
    context_builder = build_context_builder(cfg)
    messages = context_builder.build("longdocurl", case)
    result = call_llm_messages(qa_model_name, messages, qa_client)

    if result is None:
        logger.warning(f"LLM returned None for question_id {case.get('question_id', 'unknown')}. Skipping this case.")
        return

    # extract concise answer
    with open(extractor_prompt_path) as f:
        extractor_prompt = f.read()
    prompt = extractor_prompt + "\nQuestion: " + question + "\nAnalysis: " + result
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
        # Keep the logging level as configured instead of forcing INFO
        with Pool(processes=workers) as pool:  # You can adjust the number of processes as needed
            list(tqdm(pool.imap(eval_per_record, tasks), total=len(tasks)))
    else:
        logger.error("Process Mode Error!")


def run_longdocurl(cfg: DictConfig):
    benchmark_cfg = cfg.benchmarks
    output_datapath = benchmark_cfg.get("results_file") or build_default_results_file(cfg, benchmark_cfg)
    os.makedirs(os.path.dirname(output_datapath), exist_ok=True)
        
    logger.info(f"Output Datapath: {output_datapath}")

    dataset = preprocess(benchmark_cfg.qa_file, output_datapath, benchmark_cfg.image_prefix)

    # try_cnt = 2
    # while try_cnt:
    #     try_cnt -= 1
    #     try:
    #         evaluate(
    #             cfg,
    #             dataset,
    #             output_datapath
    #         )
    #     except Exception as e:
    #         logger.error(f"An Error Occurred: {e}")
    #         logger.info("Restarting Script...")
    #         time.sleep(1)
    
    evaluate(cfg, dataset, output_datapath)

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
