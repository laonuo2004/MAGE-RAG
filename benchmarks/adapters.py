import ast
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Protocol

from benchmarks.longdocurl.utils.calculate_metrics import calculate_accuracy as calculate_longdocurl_accuracy
from benchmarks.longdocurl.utils.calculate_metrics_fine_grained import calculate_accuracy_fine_grained, generalize_score_dict
from benchmarks.longdocurl.utils.utils_score_v3 import eval_score as eval_longdocurl_score
from benchmarks.mmlongbench.eval.eval_score import eval_acc_and_f1, eval_score as eval_mmlongbench_score
from utils.config_utils import get_config_value, require_config_value
from utils.llm_utils import call_llm_messages, completion_content, text_content_parts
from utils.serialization_utils import to_plain_data

logger = logging.getLogger(__name__)

MMLONGBENCH_ROOT = Path(__file__).resolve().parent / "mmlongbench"
LONGDOCURL_ROOT = Path(__file__).resolve().parent / "longdocurl"
MMLONGBENCH_EXTRACTOR_PROMPT = MMLONGBENCH_ROOT / "eval" / "prompt_for_answer_extraction.md"
LONGDOCURL_EXTRACTOR_PROMPT = LONGDOCURL_ROOT / "eval" / "prompt_for_answer_extraction.md"
LONGDOCURL_SCORE_SAMPLE_FILE = LONGDOCURL_ROOT / "evaluation_results" / "scores_sample_fine_grained.json"
COMMON_RESULT_FIELDS = (
    "prepare_metadata",
    "generation_metadata",
    "extraction_metadata",
    "pred",
    "pred_format",
    "score",
)


class BenchmarkAdapter(Protocol):
    name: str

    def load_samples(self, cfg) -> List[Dict[str, Any]]:
        ...

    def sample_key(self, sample: Dict[str, Any]) -> Any:
        ...

    def is_successful_result(self, sample: Dict[str, Any]) -> bool:
        ...

    def process_sample(self, sample: Dict[str, Any], cfg, context_builder, client) -> Dict[str, Any] | None:
        ...

    def build_metrics(self, samples: List[Dict[str, Any]], output_path: Path) -> Dict[str, Any]:
        ...


def _reset_sample_fields(sample: Dict[str, Any], keys) -> None:
    for key in keys:
        sample.pop(key, None)


def _completion_metadata(completion: Any, model_name: str) -> Dict[str, Any]:
    if isinstance(completion, str):
        return {"model": model_name}
    choice = completion.choices[0] if getattr(completion, "choices", None) else None
    metadata = {
        "id": getattr(completion, "id", None),
        "model": getattr(completion, "model", None) or model_name,
        "created": getattr(completion, "created", None),
        "finish_reason": getattr(choice, "finish_reason", None) if choice is not None else None,
        "usage": to_plain_data(getattr(completion, "usage", None)),
    }
    return {key: value for key, value in metadata.items() if value is not None}


def _prepare_messages(
    sample: Dict[str, Any],
    cfg,
    context_builder,
    benchmark_name: str,
) -> tuple[Any, str, str]:
    prep_start = perf_counter()
    messages = context_builder.build(benchmark_name, sample)
    prepare_metadata = to_plain_data(getattr(messages, "metadata", None)) or {}
    prepare_metadata["duration_seconds"] = round(perf_counter() - prep_start, 3)
    sample["prepare_metadata"] = prepare_metadata
    qa_model_name = require_config_value(cfg, "benchmarks.qa_model_name")
    extractor_model_name = require_config_value(cfg, "benchmarks.extractor_model_name")
    return messages, qa_model_name, extractor_model_name


def _generate_response(
    sample: Dict[str, Any],
    messages,
    model_name: str,
    client,
    log_prefix: str,
) -> str:
    generation_start = perf_counter()
    completion = call_llm_messages(
        client,
        model_name,
        messages,
        max_tokens=8192,
        temperature=0.0,
        retries=3,
        logger=logger,
        log_prefix=log_prefix,
        failure_value=lambda exc: f"Failed: {exc}",
    )
    response = completion_content(completion)
    sample["generation_metadata"] = {
        **_completion_metadata(completion, model_name),
        "response": response,
        "duration_seconds": round(perf_counter() - generation_start, 3),
    }
    return response


def _extract_prediction(
    sample: Dict[str, Any],
    messages,
    model_name: str,
    client,
    parse_extraction_result,
    log_prefix: str,
) -> tuple[Any, str | None]:
    extraction_start = perf_counter()
    completion = call_llm_messages(
        client,
        model_name,
        messages,
        max_tokens=8192,
        temperature=0.0,
        retries=3,
        logger=logger,
        log_prefix=log_prefix,
        failure_value=lambda exc: f"Failed: {exc}",
    )
    extracted_res = completion_content(completion)
    sample["extraction_metadata"] = {
        **_completion_metadata(completion, model_name),
        "extracted_res": extracted_res,
        "duration_seconds": round(perf_counter() - extraction_start, 3),
    }
    return parse_extraction_result(extracted_res)


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
    def parse_extraction_result(extracted_res: Any) -> tuple[str | None, str | None]:
        text = str(extracted_res or "")
        answer_match = re.search(r"Extracted answer:\s*(.*?)(?:\n+Answer format:|$)", text, flags=re.DOTALL)
        if not answer_match:
            return None, None
        format_match = re.search(r"Answer format:\s*(.*?)(?:\n|$)", text, flags=re.DOTALL)
        pred_format = format_match.group(1).strip() if format_match else None
        return answer_match.group(1).strip(), pred_format

    def build_extraction_messages(self, sample: Dict[str, Any], response: Any) -> List[Dict[str, Any]]:
        question = "" if sample.get("question") is None else str(sample.get("question"))
        output = "" if response is None else str(response)
        prompt = (
            self.extractor_prompt
            + "\n\nQuestion: "
            + question
            + "\nAnalysis: "
            + output
        )
        return [{"role": "user", "content": text_content_parts(prompt)}]

    def process_sample(self, sample: Dict[str, Any], cfg, context_builder, client) -> Dict[str, Any] | None:
        sample = dict(sample)
        _reset_sample_fields(sample, COMMON_RESULT_FIELDS)

        messages, qa_model_name, extractor_model_name = _prepare_messages(
            sample,
            cfg,
            context_builder,
            "mmlongbench",
        )
        response = _generate_response(sample, messages, qa_model_name, client, "MMLongBench generation")

        extractor_messages = self.build_extraction_messages(sample, response)
        pred, pred_format = _extract_prediction(
            sample,
            extractor_messages,
            extractor_model_name,
            client,
            self.parse_extraction_result,
            "MMLongBench extraction",
        )
        if pred is None:
            logger.warning("Failed to extract MMLongBench answer. doc_id=%s", sample.get("doc_id"))
            return None
        sample["pred"] = pred
        sample["pred_format"] = pred_format
        sample["score"] = eval_mmlongbench_score(sample["answer"], pred, sample["answer_format"])
        return sample

    def build_metrics(self, samples: List[Dict[str, Any]], output_path: Path) -> Dict[str, Any]:
        completed = [sample for sample in samples if self.is_successful_result(sample)]
        overall_acc, overall_f1 = eval_acc_and_f1(completed)
        metrics: Dict[str, Any] = {
            "overall_acc": overall_acc,
            "overall_f1": overall_f1,
        }

        def group_metrics(group_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
            acc, f1 = eval_acc_and_f1(group_samples)
            return {"acc": acc, "f1": f1, "count": len(group_samples)}

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
            "single_page": group_metrics(single_page),
            "cross_page": group_metrics(cross_page),
            "unanswerable": group_metrics(unanswerable),
        }

        evidence_source_samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        document_type_samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        answer_format_samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for sample in completed:
            for source in _ensure_list(sample.get("evidence_sources")):
                evidence_source_samples[str(source)].append(sample)
            document_type_samples[str(sample.get("doc_type", "unknown"))].append(sample)
            answer_format_samples[str(sample.get("answer_format", "unknown"))].append(sample)

        metrics["evidence_source_breakdowns"] = {
            key: group_metrics(group_samples)
            for key, group_samples in sorted(evidence_source_samples.items())
        }
        metrics["document_type_breakdowns"] = {
            key: group_metrics(group_samples)
            for key, group_samples in sorted(document_type_samples.items())
        }
        metrics["answer_format_breakdowns"] = {
            key: group_metrics(group_samples)
            for key, group_samples in sorted(answer_format_samples.items())
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

    def build_extraction_messages(self, sample: Dict[str, Any], response: Any) -> List[Dict[str, Any]]:
        prompt = f"{self.extractor_prompt}\nQuestion: {sample['question']}\nAnalysis: {response}"
        return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    def parse_extraction_result(self, extracted_res: Any) -> tuple[Any, str | None]:
        try:
            concise_answer = re.findall(r"<concise_answer>(.*?)</concise_answer>", str(extracted_res), re.DOTALL)[0]
        except Exception:
            return None, None
        format_match = re.search(r"<answer_format>(.*?)</answer_format>", str(extracted_res), re.DOTALL)
        pred_format = format_match.group(1).strip() if format_match else None
        return self.parse_concise_answer(concise_answer), pred_format

    def process_sample(self, sample: Dict[str, Any], cfg, context_builder, client) -> Dict[str, Any] | None:
        sample = dict(sample)
        _reset_sample_fields(sample, COMMON_RESULT_FIELDS)

        messages, qa_model_name, extractor_model_name = _prepare_messages(
            sample,
            cfg,
            context_builder,
            "longdocurl",
        )
        response = _generate_response(sample, messages, qa_model_name, client, "LongDocURL generation")
        if response is None:
            return None

        extractor_messages = self.build_extraction_messages(sample, response)
        pred, pred_format = _extract_prediction(
            sample,
            extractor_messages,
            extractor_model_name,
            client,
            self.parse_extraction_result,
            "LongDocURL extraction",
        )
        if pred is None:
            logger.warning("Failed to extract LongDocURL answer. question_id=%s", sample.get("question_id"))
            return None
        sample["pred"] = pred
        sample["pred_format"] = pred_format
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
