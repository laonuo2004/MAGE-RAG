import ast
import json
import logging
import re
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Protocol

from baselines.wrapper import build_context_builder
from benchmarks.longdocurl.eval.utils_api import get_msg_format
from benchmarks.longdocurl.utils.calculate_metrics import calculate_accuracy as calculate_longdocurl_accuracy
from benchmarks.longdocurl.utils.calculate_metrics_fine_grained import calculate_accuracy_fine_grained, generalize_score_dict
from benchmarks.longdocurl.utils.utils_score_v3 import eval_score as eval_longdocurl_score
from benchmarks.mmlongbench.eval.eval_score import eval_acc_and_f1, eval_score as eval_mmlongbench_score
from benchmarks.mmlongbench.eval.extract_answer import extract_answer
from utils.config_utils import get_config_value, require_config_value
from utils.llm_utils import build_openai_client, call_llm_messages

logger = logging.getLogger(__name__)

MMLONGBENCH_ROOT = Path(__file__).resolve().parent / "mmlongbench"
LONGDOCURL_ROOT = Path(__file__).resolve().parent / "longdocurl"
MMLONGBENCH_EXTRACTOR_PROMPT = MMLONGBENCH_ROOT / "eval" / "prompt_for_answer_extraction.md"
LONGDOCURL_EXTRACTOR_PROMPT = LONGDOCURL_ROOT / "eval" / "prompt_for_answer_extraction.md"
LONGDOCURL_SCORE_SAMPLE_FILE = LONGDOCURL_ROOT / "evaluation_results" / "scores_sample_fine_grained.json"


class BenchmarkAdapter(Protocol):
    name: str

    def load_samples(self, cfg) -> List[Dict[str, Any]]:
        ...

    def sample_key(self, sample: Dict[str, Any]) -> Any:
        ...

    def is_successful_result(self, sample: Dict[str, Any]) -> bool:
        ...

    def process_sample(self, sample: Dict[str, Any], cfg) -> Dict[str, Any] | None:
        ...

    def build_metrics(self, samples: List[Dict[str, Any]], output_path: Path) -> Dict[str, Any]:
        ...


def _merge_context_metadata(sample: Dict[str, Any], messages) -> None:
    metadata = getattr(messages, "metadata", None)
    if not metadata:
        return
    existing = sample.get("context_metadata")
    if not isinstance(existing, dict):
        existing = {}
    sample["context_metadata"] = {**existing, **metadata}


def _request_llm(messages, model_name: str, client, log_prefix: str, max_tokens: int = 1024) -> str:
    return call_llm_messages(
        client,
        model_name,
        messages,
        max_tokens=max_tokens,
        temperature=0.0,
        retries=10,
        logger=logger,
        log_prefix=log_prefix,
        failure_value=lambda exc: f"Failed: {exc}",
    )


class MMLongBenchAdapter:
    name = "mmlongbench"

    def __init__(self) -> None:
        self.extractor_prompt = MMLONGBENCH_EXTRACTOR_PROMPT.read_text(encoding="utf-8")

    def load_samples(self, cfg) -> List[Dict[str, Any]]:
        input_path = require_config_value(cfg, "benchmarks.input_path")
        with open(input_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        logger.info("Loaded %s MMLongBench samples from %s", len(samples), input_path)
        return samples

    def sample_key(self, sample: Dict[str, Any]) -> Any:
        return (
            sample.get("doc_id"),
            sample.get("question"),
            sample.get("answer"),
            sample.get("answer_format"),
        )

    def is_successful_result(self, sample: Dict[str, Any]) -> bool:
        return "score" in sample and sample.get("pred") != "Failed to extract"

    @staticmethod
    def parse_extracted_answer(extracted_res: Any) -> str | None:
        text = str(extracted_res or "")
        match = re.search(r"Extracted answer:\s*(.*?)(?:\n+Answer format:|$)", text, flags=re.DOTALL)
        if not match:
            return None
        return match.group(1).strip()

    def process_sample(self, sample: Dict[str, Any], cfg) -> Dict[str, Any] | None:
        sample = dict(sample)
        for key in ("score", "pred", "extracted_res", "error", "failure_stage", "status"):
            sample.pop(key, None)

        prep_start = perf_counter()
        context_builder = build_context_builder(cfg)
        messages = context_builder.build("mmlongbench", sample)
        _merge_context_metadata(sample, messages)
        client = build_openai_client(cfg)
        qa_model_name = require_config_value(cfg, "benchmarks.qa_model_name")
        extractor_model_name = require_config_value(cfg, "benchmarks.extractor_model_name")
        sample["timing_prepare_seconds"] = round(perf_counter() - prep_start, 3)

        generation_start = perf_counter()
        response = _request_llm(messages, qa_model_name, client, "MMLongBench generation")
        sample["timing_generation_seconds"] = round(perf_counter() - generation_start, 3)
        sample["response"] = response

        extraction_start = perf_counter()
        extracted_res = extract_answer(
            sample["question"],
            response,
            self.extractor_prompt,
            model_name=extractor_model_name,
            client=client,
        )
        sample["timing_extraction_seconds"] = round(perf_counter() - extraction_start, 3)
        sample["extractor_model_name"] = extractor_model_name
        sample["extracted_res"] = extracted_res
        pred = self.parse_extracted_answer(extracted_res)
        if pred is None:
            logger.warning("Failed to extract MMLongBench answer. doc_id=%s", sample.get("doc_id"))
            return None
        sample["pred"] = pred
        sample["score"] = eval_mmlongbench_score(sample["answer"], pred, sample["answer_format"])
        return sample

    def build_metrics(self, samples: List[Dict[str, Any]], output_path: Path) -> Dict[str, Any]:
        completed = [sample for sample in samples if self.is_successful_result(sample)]
        overall_acc, overall_f1 = eval_acc_and_f1(completed)
        metrics: Dict[str, Any] = {
            "overall_acc": overall_acc,
            "overall_f1": overall_f1,
        }
        single_page = [
            sample for sample in completed
            if len(_ensure_list(sample.get("evidence_pages"))) == 1
        ]
        cross_page = [
            sample for sample in completed
            if len(_ensure_list(sample.get("evidence_pages"))) != 1 and sample.get("answer") != "Not answerable"
        ]
        unanswerable = [sample for sample in completed if sample.get("answer") == "Not answerable"]
        metrics["breakdowns"] = {
            "single_page": {"acc": eval_acc_and_f1(single_page)[0], "count": len(single_page)},
            "cross_page": {"acc": eval_acc_and_f1(cross_page)[0], "count": len(cross_page)},
            "unanswerable": {"acc": eval_acc_and_f1(unanswerable)[0], "count": len(unanswerable)},
        }
        return metrics


class LongDocURLAdapter:
    name = "longdocurl"

    def __init__(self) -> None:
        self.extractor_prompt = LONGDOCURL_EXTRACTOR_PROMPT.read_text(encoding="utf-8")

    def load_samples(self, cfg) -> List[Dict[str, Any]]:
        qa_file = require_config_value(cfg, "benchmarks.qa_file")
        image_prefix = get_config_value(cfg, "benchmarks.image_prefix")
        samples = []
        with open(qa_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                sample = json.loads(line)
                sample.setdefault("question_id", idx)
                if image_prefix is not None:
                    images = []
                    for image_path in sample.get("images", []):
                        images.append(str(Path(image_prefix) / "/".join(str(image_path).split("/")[-2:])))
                    sample["images"] = images
                samples.append(sample)
        logger.info("Loaded %s LongDocURL samples from %s", len(samples), qa_file)
        return samples

    def sample_key(self, sample: Dict[str, Any]) -> Any:
        return sample.get("question_id")

    def is_successful_result(self, sample: Dict[str, Any]) -> bool:
        return "score" in sample and sample.get("pred") != "Fail to extract"

    @staticmethod
    def parse_concise_answer(concise_answer: Any) -> Any:
        text = str(concise_answer).strip()
        try:
            parsed_answer = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text
        if isinstance(parsed_answer, set):
            return list(parsed_answer)
        return parsed_answer

    def process_sample(self, sample: Dict[str, Any], cfg) -> Dict[str, Any] | None:
        sample = dict(sample)
        for key in ("score", "score_v3", "pred", "detailed_response", "response", "extracted_res"):
            sample.pop(key, None)

        prep_start = perf_counter()
        context_builder = build_context_builder(cfg)
        messages = context_builder.build("longdocurl", sample)
        _merge_context_metadata(sample, messages)
        client = build_openai_client(cfg)
        qa_model_name = require_config_value(cfg, "benchmarks.qa_model_name")
        extractor_model_name = require_config_value(cfg, "benchmarks.extractor_model_name")
        sample["timing_prepare_seconds"] = round(perf_counter() - prep_start, 3)

        generation_start = perf_counter()
        response = _request_llm(messages, qa_model_name, client, "LongDocURL generation")
        sample["timing_generation_seconds"] = round(perf_counter() - generation_start, 3)
        if response is None:
            return None
        sample["response"] = response

        prompt = f"{self.extractor_prompt}\nQuestion: {sample['question']}\nAnalysis: {response}"
        extraction_start = perf_counter()
        extractor_messages = get_msg_format(prompt, None)
        extractor_result = _request_llm(
            extractor_messages,
            extractor_model_name,
            client,
            "LongDocURL extraction",
            max_tokens=32768,
        )
        sample["timing_extraction_seconds"] = round(perf_counter() - extraction_start, 3)
        sample["extractor_model_name"] = extractor_model_name
        sample["extracted_res"] = extractor_result

        try:
            concise_answer = re.findall(r"<concise_answer>(.*?)</concise_answer>", extractor_result, re.DOTALL)[0]
        except Exception:
            logger.warning("Failed to extract LongDocURL answer. question_id=%s", sample.get("question_id"))
            return None
        pred = self.parse_concise_answer(concise_answer)
        sample["pred"] = pred
        sample["score"] = eval_longdocurl_score(sample["answer"], pred, sample["answer_format"])
        return sample

    def build_metrics(self, samples: List[Dict[str, Any]], output_path: Path) -> Dict[str, Any]:
        completed = [sample for sample in samples if self.is_successful_result(sample)]
        if completed:
            avg_acc = calculate_longdocurl_accuracy(
                [sample["pred"] for sample in completed],
                [sample["answer"] for sample in completed],
                [sample["answer_format"] for sample in completed],
            )
        else:
            avg_acc = 0.0
        metrics: Dict[str, Any] = {
            "overall_acc": avg_acc,
            "avg_acc": avg_acc,
            "rectified_avg_acc": avg_acc * len(completed) / 2325,
        }
        try:
            with open(LONGDOCURL_SCORE_SAMPLE_FILE, "r", encoding="utf-8") as f:
                score_sample = json.load(f)
            score_dict = score_sample["scores"]
            sample_cnt_dict = score_sample["sample_cnt"]
            fine_grained_samples = [dict(sample) for sample in completed]
            score_dict = calculate_accuracy_fine_grained(fine_grained_samples, score_dict)
            generalize_score_dict(score_dict, sample_cnt_dict)
            metrics["fine_grained_metrics"] = score_dict
        except Exception as exc:
            logger.warning("Failed to build LongDocURL fine-grained metrics: %s", exc)
            metrics["fine_grained_metrics_error"] = str(exc)
        return metrics


def _ensure_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None or value == "":
        return []
    try:
        parsed = ast.literal_eval(value)
    except Exception:
        return [value]
    return parsed if isinstance(parsed, list) else [parsed]
