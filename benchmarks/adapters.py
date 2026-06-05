import ast
import json
import logging
import re
from collections import defaultdict
from math import isclose
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Protocol

from benchmarks.utils.data_utils import load_longdocurl_samples, load_mmlongbench_samples
from utils.config_utils import get_config_value, require_config_value
from utils.llm_utils import call_llm_messages, completion_content, text_content_parts
from utils.serialization_utils import to_plain_data

logger = logging.getLogger(__name__)

MMLONGBENCH_ROOT = Path(__file__).resolve().parent / "mmlongbench"
LONGDOCURL_ROOT = Path(__file__).resolve().parent / "longdocurl"
MMLONGBENCH_EXTRACTOR_PROMPT = MMLONGBENCH_ROOT / "data" / "metadata" / "prompt_for_answer_extraction.md"
LONGDOCURL_EXTRACTOR_PROMPT = LONGDOCURL_ROOT / "data" / "metadata" / "prompt_for_answer_extraction.md"
LONGDOCURL_SCORE_SAMPLE_FILE = LONGDOCURL_ROOT / "data" / "metadata" / "scores_sample_fine_grained.json"
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


def _read_prompt(path: Path, fallback: Path) -> str:
    prompt_path = path if path.exists() else fallback
    return prompt_path.read_text(encoding="utf-8")


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


def _prepare_messages(sample: Dict[str, Any], cfg, context_builder, benchmark_name: str, client) -> tuple[Any, str, str]:
    prep_start = perf_counter()
    messages = context_builder.build(benchmark_name, sample, client=client)
    prepare_metadata = to_plain_data(getattr(messages, "metadata", None)) or {}
    prepare_metadata["duration_seconds"] = round(perf_counter() - prep_start, 3)
    sample["prepare_metadata"] = prepare_metadata
    qa_model_name = require_config_value(cfg, "benchmarks.qa_model_name")
    extractor_model_name = require_config_value(cfg, "benchmarks.extractor_model_name")
    return messages, qa_model_name, extractor_model_name

# g2-reader 专用旁路
def _process_self_answering_sample(
    sample: Dict[str, Any],
    context_builder,
    benchmark_name: str,
    client,
) -> Dict[str, Any] | None:
    if not hasattr(context_builder, "run_sample"):
        return None
    run_start = perf_counter()
    payload = context_builder.run_sample(benchmark_name, sample, client=client)
    if payload is None:
        return None
    metadata = dict(payload.get("metadata") or {})
    metadata["duration_seconds"] = round(perf_counter() - run_start, 3)
    sample["prepare_metadata"] = metadata
    sample["generation_metadata"] = {
        "model": metadata.get("model"),
        "response": payload.get("response"),
        "duration_seconds": metadata["duration_seconds"],
    }
    if payload.get("usage") is not None:
        sample["generation_metadata"]["usage"] = payload.get("usage")
    sample["pred"] = payload.get("pred")
    sample["pred_format"] = payload.get("pred_format")
    return sample


def _generate_response(sample: Dict[str, Any], messages, model_name: str, client, log_prefix: str) -> str:
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


def _extract_prediction(sample: Dict[str, Any], messages, model_name: str, client, parse_extraction_result, log_prefix: str) -> tuple[Any, str | None]:
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


def _levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def _anls_compute(groundtruth: Any, prediction: Any, threshold: float = 0.5) -> float:
    groundtruth = str(groundtruth)
    prediction = str(prediction)
    dist = _levenshtein_distance(groundtruth, prediction)
    length = max(len(groundtruth.upper()), len(prediction.upper()))
    value = 0.0 if length == 0 else float(dist) / float(length)
    anls = 1.0 - value
    return 0.0 if anls <= threshold else anls


def _is_float_equal(reference: Any, prediction: Any, include_percentage: bool = False, is_close: bool = False) -> bool:
    def get_precision(value: float) -> int:
        precision = 3
        if "." in str(value):
            precision = len(str(value).split(".")[-1])
        return precision

    reference = float(str(reference).strip().rstrip("%").strip())
    try:
        prediction = float(str(prediction).strip().rstrip("%").strip())
    except Exception:
        return False

    candidates = [reference / 100, reference, reference * 100] if include_percentage else [reference]
    for item in candidates:
        try:
            if is_close and isclose(item, prediction, rel_tol=0.01):
                return True
            precision = max(min(get_precision(prediction), get_precision(item)), 2)
            if round(prediction, precision) == round(item, precision):
                return True
        except Exception:
            continue
    return False


def _is_exact_match(text: str) -> bool:
    if "https://" in text:
        return True
    if text.endswith(".py") or text.endswith("ipynb"):
        return True
    if text.startswith("page"):
        return True
    if re.fullmatch(r"\b\d+(-\d+|\s\d+)?\b", text):
        return True
    if "a.m." in text or "p.m." in text:
        return True
    if re.fullmatch(r"\b\d{4}[-\s]\d{2}[-\s]\d{2}\b", text):
        return True
    if re.fullmatch(r"\b\d{4}[-\s]\d{2}\b", text):
        return True
    if re.fullmatch(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text):
        return True
    return False


def _is_float_like(value: Any) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def _maybe_parse_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{(":
        return value
    try:
        return ast.literal_eval(text)
    except Exception:
        return value


class MMLongBenchAdapter:
    name = "mmlongbench"

    def __init__(self) -> None:
        self.extractor_prompt = _read_prompt(
            MMLONGBENCH_EXTRACTOR_PROMPT,
            MMLONGBENCH_ROOT / "eval" / "prompt_for_answer_extraction.md",
        )

    @staticmethod
    def _clean_string(value: Any) -> str:
        text = str(value).lower().strip()
        for suffix in ("mile", "miles", "million"):
            if text.endswith(suffix):
                text = text.removesuffix(suffix).strip()
        text = re.sub(r"\s*\([^)]*\)", "", text).strip()
        text = re.sub(r"^['\"]|['\"]$", "", text).strip()
        return text.strip().lstrip("$").strip().rstrip("%").strip()

    @staticmethod
    def score(gt: Any, pred: Any, answer_type: str) -> float:
        if answer_type == "Int":
            try:
                gt = int(float(MMLongBenchAdapter._clean_string(gt)))
                pred = int(float(MMLongBenchAdapter._clean_string(pred)))
            except Exception:
                pred = ""
            score = gt == pred
        elif answer_type == "Float":
            try:
                gt = float(MMLongBenchAdapter._clean_string(gt))
                pred = float(MMLongBenchAdapter._clean_string(pred))
            except Exception:
                pred = ""
            score = _is_float_equal(gt, pred, include_percentage=True, is_close=True)
        elif answer_type in ["Str", "String", "None"]:
            gt = MMLongBenchAdapter._clean_string(gt)
            pred = MMLongBenchAdapter._clean_string(pred)
            score = gt == pred if _is_exact_match(gt) else _anls_compute(gt, pred)
        else:
            gt = _maybe_parse_literal(gt)
            gt = gt if isinstance(gt, list) else [gt]
            pred = _maybe_parse_literal(pred)
            pred = pred if isinstance(pred, list) else [pred]
            if len(gt) != len(pred):
                score = 0.0
            else:
                gt = sorted([MMLongBenchAdapter._clean_string(item) for item in gt])
                pred = sorted([MMLongBenchAdapter._clean_string(item) for item in pred])
                if not gt:
                    score = 0.0
                elif _is_float_like(gt[0]) or _is_exact_match(gt[0]):
                    score = "-".join(gt) == "-".join(pred)
                else:
                    score = min([_anls_compute(gt_v, pred_v) for gt_v, pred_v in zip(gt, pred)])
        return float(score)

    @staticmethod
    def acc_and_f1(samples: List[Dict[str, Any]]) -> tuple[float, float]:
        evaluated_samples = [sample for sample in samples if "score" in sample]
        if not evaluated_samples:
            return 0.0, 0.0

        acc = sum([sample["score"] for sample in evaluated_samples]) / len(evaluated_samples)
        try:
            answerable = [sample for sample in evaluated_samples if sample["answer"] != "Not answerable"]
            predicted_answerable = [sample for sample in evaluated_samples if sample["pred"] != "Not answerable"]
            recall = sum([sample["score"] for sample in answerable]) / len(answerable)
            precision = sum([sample["score"] for sample in answerable]) / len(predicted_answerable)
            f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0.0 else 0.0
        except Exception:
            f1 = 0.0
        return acc, f1

    def load_samples(self, cfg) -> List[Dict[str, Any]]:
        input_path = require_config_value(cfg, "benchmarks.input_path")
        samples = load_mmlongbench_samples(input_path)
        logger.info("Loaded %s MMLongBench samples from %s", len(samples), input_path)
        return samples

    def sample_key(self, sample: Dict[str, Any]) -> Any:
        return (sample.get("doc_id"), sample.get("question"), sample.get("answer"), sample.get("answer_format"))

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

    def build_extraction_messages(
        self,
        sample: Dict[str, Any],
        response: Any,
    ) -> List[Dict[str, Any]]:
        question = "" if sample.get("question") is None else str(sample.get("question"))
        output = "" if response is None else str(response)
        prompt = self.extractor_prompt + "\n\nQuestion: " + question + "\nAnalysis: " + output
        return [{"role": "user", "content": text_content_parts(prompt)}]

    def process_sample(self, sample: Dict[str, Any], cfg, context_builder, client) -> Dict[str, Any] | None:
        sample = dict(sample)
        _reset_sample_fields(sample, COMMON_RESULT_FIELDS)
        self_answered = _process_self_answering_sample(sample, context_builder, "mmlongbench", client)
        if self_answered is not None:
            if self_answered.get("pred") is None:
                return None
            self_answered["score"] = self.score(
                self_answered["answer"],
                self_answered["pred"],
                self_answered["answer_format"],
            )
            return self_answered
        messages, qa_model_name, extractor_model_name = _prepare_messages(sample, cfg, context_builder, "mmlongbench", client)
        response = _generate_response(sample, messages, qa_model_name, client, "MMLongBench generation")
        pred, pred_format = _extract_prediction(
            sample,
            self.build_extraction_messages(sample, response),
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
        sample["score"] = self.score(sample["answer"], pred, sample["answer_format"])
        return sample

    def build_metrics(self, samples: List[Dict[str, Any]], output_path: Path) -> Dict[str, Any]:
        completed = [sample for sample in samples if self.is_successful_result(sample)]
        overall_acc, overall_f1 = self.acc_and_f1(completed)
        metrics: Dict[str, Any] = {"overall_acc": overall_acc, "overall_f1": overall_f1}

        def group_metrics(group_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
            acc, f1 = self.acc_and_f1(group_samples)
            return {"acc": acc, "f1": f1, "count": len(group_samples)}

        single_page = [sample for sample in completed if len(_ensure_list(sample.get("evidence_pages"))) == 1]
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
            key: group_metrics(group_samples) for key, group_samples in sorted(evidence_source_samples.items())
        }
        metrics["document_type_breakdowns"] = {
            key: group_metrics(group_samples) for key, group_samples in sorted(document_type_samples.items())
        }
        metrics["answer_format_breakdowns"] = {
            key: group_metrics(group_samples) for key, group_samples in sorted(answer_format_samples.items())
        }
        return metrics


class LongDocURLAdapter:
    name = "longdocurl"

    def __init__(self) -> None:
        self.extractor_prompt = _read_prompt(
            LONGDOCURL_EXTRACTOR_PROMPT,
            LONGDOCURL_ROOT / "eval" / "prompt_for_answer_extraction.md",
        )


    @staticmethod
    def _clean_string(value: Any) -> str:
        text = str(value).lower().strip().replace(",", "")
        for suffix in ("kg", "mm", "m", "meters", "acres", "minutes", "mile", "miles", "million", "thousand", "billion"):
            if text.endswith(suffix):
                text = text.removesuffix(suffix).strip()
        text = re.sub(r"\s*\([^)]*\)", "", text).strip()
        text = re.sub(r"^['\"]|['\"]$", "", text).strip()
        return text.strip().lstrip("$").strip().lstrip("£").strip().rstrip("%").strip()

    @staticmethod
    def _parse_answer_list(value: Any) -> list[Any]:
        if isinstance(value, str) and value.startswith("["):
            try:
                value = ast.literal_eval(value)
            except Exception:
                pass
        if not isinstance(value, list):
            value = [value]
        if value and isinstance(value[0], dict):
            value = ["-".join([str(item_value) for item_value in item.values()]) for item in value]
        return value

    @staticmethod
    def score(gt: Any, pred: Any, answer_type: str) -> float:
        if answer_type == "Integer":
            try:
                gt_clean = LongDocURLAdapter._clean_string(gt)
                gt = int(gt_clean)
            except Exception:
                pass
            try:
                pred_clean = LongDocURLAdapter._clean_string(pred)
                pred = int(pred_clean)
            except Exception:
                pred = ""
            score: float | bool = gt == pred
        elif answer_type == "Float":
            gt_clean = LongDocURLAdapter._clean_string(gt)
            pred_clean = LongDocURLAdapter._clean_string(pred)
            try:
                gt_value: Any = float(gt_clean)
            except Exception:
                gt_value = gt_clean
            try:
                pred_value: Any = float(pred_clean)
            except Exception:
                pred_value = str(pred_clean)
            try:
                score = _is_float_equal(gt_value, pred_value, include_percentage=True, is_close=True)
            except Exception:
                score = 0.0
        elif answer_type in ["String", "Str", "None"]:
            gt_clean = LongDocURLAdapter._clean_string(gt)
            pred_clean = LongDocURLAdapter._clean_string(pred)
            score = gt_clean == pred_clean if _is_exact_match(gt_clean) else _anls_compute(gt_clean, pred_clean)
        else:
            gt_list = LongDocURLAdapter._parse_answer_list(gt)
            pred_list = LongDocURLAdapter._parse_answer_list(pred)
            if not pred_list:
                return 0.0
            gt_clean = [LongDocURLAdapter._clean_string(item) for item in gt_list]
            pred_clean = [LongDocURLAdapter._clean_string(item) for item in pred_list]
            if not gt_clean:
                score = 0.0
            elif _is_float_like(gt_clean[0]) or _is_exact_match(gt_clean[0]):
                score = "-".join(gt_clean) == "-".join(pred_clean)
            else:
                greedy_scores = [max([_anls_compute(str(gt_v), str(pred_v)) for pred_v in pred_clean]) for gt_v in gt_clean]
                score = sum(greedy_scores) / len(gt_clean) * min(1, len(gt_clean) / len(pred_clean)) ** 0.5
        return float(score)

    @staticmethod
    def accuracy(answers: list[Any], annotations: list[Any], answer_formats: list[str]) -> float:
        if not answers:
            return 0.0
        total_scores = 0.0
        for pred_ans, annotation, answer_format in zip(answers, annotations, answer_formats):
            score = 0.0 if pred_ans == "Fail to extract" else LongDocURLAdapter.score(annotation, pred_ans, answer_format)
            total_scores += score
        return total_scores / len(answers)

    @staticmethod
    def accuracy_fine_grained(samples: List[Dict[str, Any]], score_dict: Dict[str, Any]) -> Dict[str, Any]:
        for sample in samples:
            pred_ans, annotation, answer_format = sample["pred"], sample["answer"], sample["answer_format"]
            sample["score_v3"] = 0.0 if pred_ans == "Fail to extract" else LongDocURLAdapter.score(annotation, pred_ans, answer_format)

        for sample in samples:
            score_dict["Main_Task"][sample["task_tag"]] += sample["score_v3"]
            for evidence_source in sample["evidence_sources"]:
                if evidence_source in ["Text", "Layout", "Figure", "Table"]:
                    score_dict["Element_Type"][evidence_source] += sample["score_v3"]
            if len(sample["evidence_pages"]) > 1:
                score_dict["Evidence_Pages"]["Multi_Page"] += sample["score_v3"]
            elif len(sample["evidence_pages"]) == 1:
                score_dict["Evidence_Pages"]["Single_Page"] += sample["score_v3"]
            if len(sample["evidence_sources"]) > 1:
                score_dict["Num_of_Element_Types"]["Cross_Element"] += sample["score_v3"]

        for sample in samples:
            sub_score_dict = score_dict["Fine_Grained"][sample["task_tag"]]
            if sample["task_tag"] in ["Understanding", "Reasoning"]:
                if len(sample["evidence_pages"]) > 1:
                    sub_sub_score_dict = sub_score_dict["Multi_Page"]
                elif len(sample["evidence_pages"]) == 1:
                    sub_sub_score_dict = sub_score_dict["Single_Page"]
                else:
                    continue
                for evidence_source in sample["evidence_sources"]:
                    if evidence_source in ["Text", "Layout", "Figure", "Table"]:
                        sub_sub_score_dict[evidence_source] += sample["score_v3"]
            elif sample["task_tag"] == "Locating":
                sub_sub_score_dict = sub_score_dict["Cross_Element"]
                if sample["question_type"] == "topic2title":
                    sub_sub_score_dict["Cross_Title"] += sample["score_v3"]
                elif sample["question_type"] == "summary2title":
                    sub_sub_score_dict["Para_Title"] += sample["score_v3"]
                elif sample["question_type"] == "summary2tab":
                    sub_sub_score_dict["Cross_Table"] += sample["score_v3"]
                elif sample["question_type"] == "extract_fig2tab":
                    sub_sub_score_dict["Figure_Table"] += sample["score_v3"]
        return score_dict

    @staticmethod
    def generalize_score_dict(score_dict: Dict[str, Any], sample_cnt_dict: Dict[str, Any]) -> None:
        for key, value in score_dict.items():
            if isinstance(value, dict):
                LongDocURLAdapter.generalize_score_dict(value, sample_cnt_dict[key])
            else:
                score_dict[key] /= sample_cnt_dict[key]

    def load_samples(self, cfg) -> List[Dict[str, Any]]:
        qa_file = require_config_value(cfg, "benchmarks.qa_file")
        image_prefix = get_config_value(cfg, "benchmarks.image_prefix")
        samples = load_longdocurl_samples(qa_file, image_prefix=image_prefix)
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
        self_answered = _process_self_answering_sample(sample, context_builder, "longdocurl", client)
        if self_answered is not None:
            if self_answered.get("pred") is None:
                return None
            self_answered["score"] = self.score(
                self_answered["answer"],
                self_answered["pred"],
                self_answered["answer_format"],
            )
            return self_answered
        messages, qa_model_name, extractor_model_name = _prepare_messages(sample, cfg, context_builder, "longdocurl", client)
        response = _generate_response(sample, messages, qa_model_name, client, "LongDocURL generation")
        if response is None:
            return None
        pred, pred_format = _extract_prediction(
            sample,
            self.build_extraction_messages(sample, response),
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
        sample["score"] = self.score(sample["answer"], pred, sample["answer_format"])
        return sample

    def build_metrics(self, samples: List[Dict[str, Any]], output_path: Path) -> Dict[str, Any]:
        completed = [sample for sample in samples if self.is_successful_result(sample)]
        avg_acc = self.accuracy(
            [sample["pred"] for sample in completed],
            [sample["answer"] for sample in completed],
            [sample["answer_format"] for sample in completed],
        )
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
            score_dict = self.accuracy_fine_grained(fine_grained_samples, score_dict)
            self.generalize_score_dict(score_dict, sample_cnt_dict)
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
