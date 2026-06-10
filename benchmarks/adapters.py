import ast
import inspect
import json
import logging
import re
from collections import defaultdict
from math import isclose
from pathlib import Path
from textwrap import dedent
from time import perf_counter
from typing import Any, Dict, List, Protocol

from benchmarks.utils.data_utils import load_longdocurl_samples, load_mmlongbench_samples
from utils.config_utils import get_config_value, require_config_value
from utils.llm_utils import call_llm_messages, completion_content, text_content_parts, xml_block
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
    "correction_metadata",
    "pred",
    "pred_format",
    "score",
    "corrected_pred",
    "corrected_format",
    "corrected_score",
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


def _finalize_result_fields(sample: Dict[str, Any]) -> Dict[str, Any]:
    correction_metadata = sample.get("correction_metadata")
    if isinstance(correction_metadata, dict):
        sample["pred"] = correction_metadata.get("initial_pred", sample.get("pred"))
        sample["pred_format"] = correction_metadata.get("initial_pred_format", sample.get("pred_format"))
        sample["score"] = correction_metadata.get("initial_score", sample.get("score"))
        sample["corrected_pred"] = correction_metadata.get("corrected_pred", sample.get("pred"))
        sample["corrected_format"] = correction_metadata.get("corrected_pred_format", sample.get("pred_format"))
        sample["corrected_score"] = correction_metadata.get("corrected_score", sample.get("score"))
    else:
        sample["corrected_pred"] = sample.get("pred")
        sample["corrected_format"] = sample.get("pred_format")
        sample["corrected_score"] = sample.get("score")

    ordered_values = {key: sample.pop(key) for key in COMMON_RESULT_FIELDS if key in sample}
    sample.update(ordered_values)
    return sample


def _final_prediction_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    final_sample = dict(sample)
    if "corrected_pred" in sample:
        final_sample["pred"] = sample.get("corrected_pred")
    if "corrected_format" in sample:
        final_sample["pred_format"] = sample.get("corrected_format")
    if "corrected_score" in sample:
        final_sample["score"] = sample.get("corrected_score")
    return final_sample


def _apply_sample_limit(samples: List[Dict[str, Any]], cfg) -> List[Dict[str, Any]]:
    limit = get_config_value(cfg, "benchmarks.limit")
    if limit is None:
        return samples
    return samples[: max(0, int(limit))]


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


def _strip_markdown_code_fence(text: Any) -> str:
    content = str(text or "").strip()
    match = re.fullmatch(r"```(?:xml|XML)?\s*(.*?)\s*```", content, flags=re.DOTALL)
    return match.group(1).strip() if match else content


def _strip_leading_think_block(text: Any) -> str:
    content = _strip_markdown_code_fence(text)
    return re.sub(r"^\s*<think>.*?</think>\s*", "", content, count=1, flags=re.DOTALL).strip()


def _unwrap_xml_block(text: Any, tag: str) -> str | None:
    content = str(text or "")
    match = re.search(rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>", content, flags=re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def _unwrap_extraction_result(text: Any) -> str:
    content = _strip_leading_think_block(text)
    extraction = _unwrap_xml_block(content, "extraction_result")
    return extraction if extraction is not None else content


def _unwrap_generation_answer(text: Any) -> str:
    content = _strip_leading_think_block(text)
    answer = _unwrap_xml_block(content, "answer")
    return answer if answer is not None else content


def _unwrap_inline_xml_value(text: Any, tags: tuple[str, ...]) -> str:
    content = _strip_markdown_code_fence(text)
    for tag in tags:
        value = _unwrap_xml_block(content, tag)
        if value is not None:
            return value
    return content.strip()


def _uses_magerag_context(sample: Dict[str, Any]) -> bool:
    prepare_metadata = sample.get("prepare_metadata")
    return isinstance(prepare_metadata, dict) and prepare_metadata.get("context_builder") == "magerag"


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
        "mode": "self_answering",
        "raw_pred": payload.get("pred"),
        "raw_pred_format": payload.get("pred_format"),
    }
    if payload.get("usage") is not None:
        sample["generation_metadata"]["usage"] = payload.get("usage")
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
    raw_response = completion_content(completion)
    response = _unwrap_generation_answer(raw_response) if _uses_magerag_context(sample) else raw_response
    sample["generation_metadata"] = {
        **_completion_metadata(completion, model_name),
        "response": response,
        "raw_response": raw_response,
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


def _format_initial_extraction(extracted_res: Any) -> str:
    return "" if extracted_res is None else str(extracted_res)


def _source_for_prompt(obj: Any) -> str | None:
    try:
        source = dedent(inspect.getsource(obj)).strip()
    except Exception:
        return None
    return source.replace("@staticmethod\n", "")


def _answer_type_family(answer_type: Any) -> str:
    normalized = str(answer_type or "").strip().lower()
    if normalized in {"int", "integer"}:
        return "int"
    if normalized == "float":
        return "float"
    if normalized in {"str", "string", "none"}:
        return "string"
    return "list"


def _score_source_snippet(adapter: Any, answer_type: Any) -> str:
    class_name = adapter.__class__.__name__
    family = _answer_type_family(answer_type)
    snippets = {
        "int": (
            "def score_int(gt: Any, pred: Any) -> float:\n"
            f"    gt = int(float({class_name}._clean_string(gt)))\n"
            f"    pred = int(float({class_name}._clean_string(pred)))\n"
            "    return float(gt == pred)"
        ),
        "float": (
            "def score_float(gt: Any, pred: Any) -> float:\n"
            f"    gt = float({class_name}._clean_string(gt))\n"
            f"    pred = float({class_name}._clean_string(pred))\n"
            "    return float(_is_float_equal(gt, pred, include_percentage=True, is_close=True))"
        ),
        "string": (
            "def score_string(gt: Any, pred: Any) -> float:\n"
            f"    gt = {class_name}._clean_string(gt)\n"
            f"    pred = {class_name}._clean_string(pred)\n"
            "    # Near-match scoring can reward semantic-preserving punctuation, case, and hyphenation differences.\n"
            "    return float(gt == pred if _is_exact_match(gt) else _anls_compute(gt, pred))"
        ),
    }
    if family != "list":
        return snippets[family]
    if class_name == "LongDocURLAdapter":
        return (
            "def score_list(gt: Any, pred: Any) -> float:\n"
            "    gt_list = LongDocURLAdapter._parse_answer_list(gt)\n"
            "    pred_list = LongDocURLAdapter._parse_answer_list(pred)\n"
            "    if not pred_list:\n"
            "        return 0.0\n"
            "    gt_clean = [LongDocURLAdapter._clean_string(item) for item in gt_list]\n"
            "    pred_clean = [LongDocURLAdapter._clean_string(item) for item in pred_list]\n"
            "    if not gt_clean:\n"
            "        return 0.0\n"
            "    if _is_float_like(gt_clean[0]) or _is_exact_match(gt_clean[0]):\n"
            "        return float('-'.join(gt_clean) == '-'.join(pred_clean))\n"
            "    greedy_scores = [max([_anls_compute(str(gt_v), str(pred_v)) for pred_v in pred_clean]) for gt_v in gt_clean]\n"
            "    return float(sum(greedy_scores) / len(gt_clean) * min(1, len(gt_clean) / len(pred_clean)) ** 0.5)"
        )
    return (
        "def score_list(gt: Any, pred: Any) -> float:\n"
        "    gt = _normalize_list_candidate(gt)\n"
        "    pred = _normalize_list_candidate(pred)\n"
        "    if len(gt) != len(pred):\n"
        "        return 0.0\n"
        "    gt = sorted([_normalize_list_item(item) for item in gt])\n"
        "    pred = sorted([_normalize_list_item(item) for item in pred])\n"
        "    if not gt:\n"
        "        return 0.0\n"
        "    if _is_float_like(gt[0]) or _is_exact_match(gt[0]):\n"
        "        return float('-'.join(gt) == '-'.join(pred))\n"
        "    return float(min([_anls_compute(gt_v, pred_v) for gt_v, pred_v in zip(gt, pred)]))"
    )


def _score_function_source(adapter: Any, answer_type: Any) -> str:
    family = _answer_type_family(answer_type)
    helpers_by_family = {
        "int": [],
        "float": [_is_float_equal],
        "string": [_levenshtein_distance, _anls_compute, _is_exact_match],
        "list": [_levenshtein_distance, _anls_compute, _is_exact_match, _is_float_like],
    }
    helpers = helpers_by_family[family]
    if family == "list":
        if adapter.__class__.__name__ == "LongDocURLAdapter":
            helpers = [adapter.__class__._parse_answer_list] + helpers
        else:
            helpers = [_maybe_parse_literal, _normalize_list_candidate, _normalize_list_item] + helpers
    helper_sources = []
    clean_source = _source_for_prompt(adapter.__class__._clean_string)
    if clean_source is not None:
        helper_sources.append(clean_source)
    for helper in helpers:
        helper_source = _source_for_prompt(helper)
        if helper_source is not None:
            helper_sources.append(helper_source)
    helper_sources.append(_score_source_snippet(adapter, answer_type))
    return "\n\n".join(helper_sources)


def _correction_output_contract(benchmark_name: str) -> str:
    if benchmark_name == "longdocurl":
        return (
            "Output sequence:\n"
            "<think>...</think>\n"
            "<corrected_extraction>\n"
            "Extracted answer: <concise_answer>[answer]</concise_answer>\n"
            "Answer format: <answer_format>[answer format]</answer_format>\n"
            "</corrected_extraction>\n"
            "No other text outside these two XML blocks."
        )
    return (
        "Output sequence:\n"
        "<think>...</think>\n"
        "<corrected_extraction>\n"
        "Extracted answer: [answer]\n"
        "Answer format: [answer format]\n"
        "</corrected_extraction>\n"
        "No other text outside these two XML blocks."
    )


def _extraction_output_contract(benchmark_name: str) -> str:
    if benchmark_name == "longdocurl":
        return (
            "Output sequence:\n"
            "<think>...</think>\n"
            "<extraction_result>\n"
            "Extracted answer: <concise_answer>[answer]</concise_answer>\n"
            "Answer format: <answer_format>[answer_format]</answer_format>\n"
            "</extraction_result>\n"
            "No other text outside these two XML blocks."
        )
    return (
        "Output sequence:\n"
        "<think>...</think>\n"
        "<extraction_result>\n"
        "Extracted answer: [answer]\n"
        "Answer format: [answer_format]\n"
        "</extraction_result>\n"
        "No other text outside these two XML blocks."
    )


def _extraction_few_shot_examples_xml(benchmark_name: str) -> str:
    if benchmark_name == "longdocurl":
        examples = [
            xml_block(
                "example",
                "\n".join([
                    xml_block("problem", "Extract a concrete integer answer.", escape=True, indent=4),
                    xml_block(
                        "example_input",
                        xml_block("question", "How many regulations were breached?", escape=True)
                        + "\n"
                        + xml_block("model_response", "The report states that 10 regulations were breached.", escape=True),
                        indent=4,
                    ),
                    xml_block(
                        "example_output",
                        "<think>The response directly gives the count. Integer is the correct format.</think>\n"
                        "<extraction_result>\n"
                        "Extracted answer: <concise_answer>10</concise_answer>\n"
                        "Answer format: <answer_format>Integer</answer_format>\n"
                        "</extraction_result>",
                        cdata=True,
                        indent=4,
                    ),
                ]),
                indent=2,
            ),
            xml_block(
                "example",
                "\n".join([
                    xml_block("problem", "Return None format only for an unanswerable final answer.", escape=True, indent=4),
                    xml_block(
                        "example_input",
                        xml_block("question", "What percentage of Chinese respondents paid more attention?", escape=True)
                        + "\n"
                        + xml_block("model_response", "The document does not specify this; the answer is Not answerable.", escape=True),
                        indent=4,
                    ),
                    xml_block(
                        "example_output",
                        "<think>The response says the requested value is not present, so the extracted answer is Not answerable and the format is None.</think>\n"
                        "<extraction_result>\n"
                        "Extracted answer: <concise_answer>Not answerable</concise_answer>\n"
                        "Answer format: <answer_format>None</answer_format>\n"
                        "</extraction_result>",
                        cdata=True,
                        indent=4,
                    ),
                ]),
                indent=2,
            ),
            xml_block(
                "example",
                "\n".join([
                    xml_block("problem", "Do not use None when a concrete answer is present.", escape=True, indent=4),
                    xml_block(
                        "example_input",
                        xml_block("question", "Wind energy percentage in 2016 is lower than that in 2017, yes or no?", escape=True)
                        + "\n"
                        + xml_block("model_response", "Yes. The 2016 wind energy percentage is lower than the 2017 percentage.", escape=True),
                        indent=4,
                    ),
                    xml_block(
                        "example_output",
                        "<think>The response gives a concrete yes/no answer. Even if the dataset has a missing format label, None is not appropriate for an answerable response.</think>\n"
                        "<extraction_result>\n"
                        "Extracted answer: <concise_answer>yes</concise_answer>\n"
                        "Answer format: <answer_format>String</answer_format>\n"
                        "</extraction_result>",
                        cdata=True,
                        indent=4,
                    ),
                ]),
                indent=2,
            ),
        ]
        return xml_block("few_shot_examples", "\n".join(examples), indent=2)
    examples = [
        xml_block(
            "example",
            "\n".join([
                xml_block("problem", "Extract a list answer from a final response.", escape=True, indent=4),
                xml_block(
                    "example_input",
                    xml_block("question", "List the primary questions asked about the services in this report.", escape=True)
                    + "\n"
                    + xml_block("model_response", "The primary questions are: Is the service safe? Is the service effective? Is the service caring? Is the service responsive? Is the service well-led?", escape=True),
                    indent=4,
                ),
                xml_block(
                    "example_output",
                    "<think>The response lists five requested questions, so the extraction should preserve all items as a List.</think>\n"
                    "<extraction_result>\n"
                    "Extracted answer: ['Is the service safe?', 'Is the service effective?', 'Is the service caring?', 'Is the service responsive?', 'Is the service well-led?']\n"
                    "Answer format: List\n"
                    "</extraction_result>",
                    cdata=True,
                    indent=4,
                ),
            ]),
            indent=2,
        ),
        xml_block(
            "example",
            "\n".join([
                xml_block("problem", "Return None format only when the answer itself is Not answerable.", escape=True, indent=4),
                xml_block(
                    "example_input",
                    xml_block("question", "Which subgroup gained the most confidence?", escape=True)
                    + "\n"
                    + xml_block("model_response", "The report does not provide enough information to answer. Final answer: Not answerable.", escape=True),
                    indent=4,
                ),
                xml_block(
                    "example_output",
                    "<think>The response explicitly concludes that the requested answer cannot be found.</think>\n"
                    "<extraction_result>\n"
                    "Extracted answer: Not answerable\n"
                    "Answer format: None\n"
                    "</extraction_result>",
                    cdata=True,
                    indent=4,
                ),
            ]),
            indent=2,
        ),
        xml_block(
            "example",
            "\n".join([
                xml_block("problem", "Prefer a concrete final answer over uncertainty in reasoning.", escape=True, indent=4),
                xml_block(
                    "example_input",
                    xml_block("question", "Which group is identified?", escape=True)
                    + "\n"
                    + xml_block("model_response", "<think>I was initially unsure, but the evidence states the group.</think><answer>less well-off</answer>", escape=True),
                    indent=4,
                ),
                xml_block(
                    "example_output",
                    "<think>The final answer block contains a concrete supported answer, so it is not Not answerable.</think>\n"
                    "<extraction_result>\n"
                    "Extracted answer: less well-off\n"
                    "Answer format: String\n"
                    "</extraction_result>",
                    cdata=True,
                    indent=4,
                ),
            ]),
            indent=2,
        ),
    ]
    return xml_block("few_shot_examples", "\n".join(examples), indent=2)


def _build_extraction_prompt(benchmark_name: str, question: Any, response: Any) -> str:
    target_formats = "Integer, Float, String, List, None"
    visibility_policy = ""
    if benchmark_name == "longdocurl":
        visibility_policy = xml_block(
            "visibility_policy",
            "If the question restricts visible pages or image ranges, discard answer components that come only from outside that requested range.",
            escape=True,
            indent=2,
        )
    input_data = "\n".join([
        xml_block("question", "" if question is None else str(question), escape=True, inline=True),
        xml_block("model_response", "" if response is None else str(response), escape=True, inline=True),
        xml_block("target_answer_formats", target_formats, inline=True),
    ])
    blocks = [
        xml_block("role", "You are a benchmark answer extraction normalizer.", escape=True, indent=2),
        xml_block(
            "objective",
            "Extract the concise benchmark answer and answer format from model_response without adding facts not present in that response.",
            escape=True,
            indent=2,
        ),
        xml_block("input_data", input_data, indent=2),
        xml_block(
            "extraction_policy",
            "Prefer the final answer span when model_response contains explicit answer markup such as <answer>...</answer>. Use reasoning text only to disambiguate that final answer. Preserve every requested item for list questions. Normalize surface form only when the normalized value is directly supported by model_response.",
            escape=True,
            indent=2,
        ),
        xml_block(
            "not_answerable_policy",
            "Use Not answerable only when model_response's final conclusion says the document evidence does not answer the question or no answer can be extracted. Do not output Not answerable merely because the reasoning mentions uncertainty before giving a concrete final answer. Use Fail to answer only when model_response says it cannot read, view, or understand the provided document/images.",
            escape=True,
            indent=2,
        ),
        xml_block(
            "format_policy",
            "Choose the most specific answer format from Integer, Float, String, or List for every concrete answer. Use None as answer format only when the extracted answer is exactly Not answerable or Fail to answer. If the model response contains a concrete answer, never use None as the answer format. For List, output a parseable list of answer items rather than a prose sentence.",
            escape=True,
            indent=2,
        ),
    ]
    if visibility_policy:
        blocks.append(visibility_policy)
    blocks.extend([
        xml_block(
            "thinking_policy",
            "First output one <think>...</think> block. In it, compare the question with model_response, identify the final answer span, decide whether the case is answerable, and verify the answer format.",
            escape=True,
            indent=2,
        ),
        xml_block("output_schema", _extraction_output_contract(benchmark_name), cdata=True, indent=2),
        xml_block(
            "self_check",
            "Before returning, verify that the extraction is supported by model_response, the answer format is one of the target formats, None is used only for Not answerable or Fail to answer, and no text appears outside the required XML blocks.",
            escape=True,
            indent=2,
        ),
        _extraction_few_shot_examples_xml(benchmark_name),
    ])
    return xml_block("extraction_prompt", "\n\n".join(blocks))


def _correction_few_shot_examples_xml(benchmark_name: str) -> str:
    contract = _correction_output_contract(benchmark_name)
    examples = [
        xml_block(
            "example",
            "\n".join([
                xml_block("problem", "Fix a formatting-only false negative.", escape=True, indent=4),
                xml_block(
                    "example_input",
                    xml_block("model_response", "The group is less well off.", escape=True)
                    + "\n"
                    + xml_block("gold_truth", "less well-off", escape=True)
                    + "\n"
                    + xml_block("initial_formatted_extraction", "Extracted answer: less well off\nAnswer format: String", escape=True),
                    indent=4,
                ),
                xml_block(
                    "example_output",
                    "<think>The response supports the gold phrase, and the mismatch is only hyphenation.</think>\n"
                    "<corrected_extraction>\n"
                    "Extracted answer: less well-off\n"
                    "Answer format: String\n"
                    "</corrected_extraction>",
                    cdata=True,
                    indent=4,
                ),
            ]),
            indent=2,
        ),
        xml_block(
            "example",
            "\n".join([
                xml_block("problem", "Keep the initial extraction when the response is substantively wrong.", escape=True, indent=4),
                xml_block(
                    "example_input",
                    xml_block("model_response", "The reported school is the French Lycee in London.", escape=True)
                    + "\n"
                    + xml_block("gold_truth", "Ecole Normale Superieure", escape=True)
                    + "\n"
                    + xml_block("initial_formatted_extraction", "Extracted answer: French Lycee in London\nAnswer format: String", escape=True),
                    indent=4,
                ),
                xml_block(
                    "example_output",
                    "<think>The model_response does not support the gold truth, so copying gold_truth would inject unsupported facts.</think>\n"
                    "<corrected_extraction>\n"
                    "Extracted answer: French Lycee in London\n"
                    "Answer format: String\n"
                    "</corrected_extraction>",
                    cdata=True,
                    indent=4,
                ),
            ]),
            indent=2,
        ),
        xml_block(
            "example",
            "\n".join([
                xml_block("problem", "Repair list serialization when all items are supported.", escape=True, indent=4),
                xml_block(
                    "example_input",
                    xml_block("model_response", "The answer is White households and 10 percentage points.", escape=True)
                    + "\n"
                    + xml_block("gold_truth", "['White', '10%']", escape=True)
                    + "\n"
                    + xml_block("initial_formatted_extraction", "Extracted answer: White households, 10 percentage points\nAnswer format: String", escape=True),
                    indent=4,
                ),
                xml_block(
                    "example_output",
                    "<think>The two list items are directly supported, and benchmark scoring expects list serialization.</think>\n"
                    "<corrected_extraction>\n"
                    "Extracted answer: ['White', '10%']\n"
                    "Answer format: List\n"
                    "</corrected_extraction>",
                    cdata=True,
                    indent=4,
                ),
            ]),
            indent=2,
        ),
    ]
    if benchmark_name == "longdocurl":
        examples[0] = examples[0].replace(
            "Extracted answer: less well-off\nAnswer format: String",
            "Extracted answer: <concise_answer>less well-off</concise_answer>\nAnswer format: <answer_format>String</answer_format>",
        )
    return xml_block(
        "few_shot_examples",
        "\n".join(examples) + "\n" + xml_block("output_contract_reference", contract, cdata=True, indent=2),
        indent=2,
    )


def _build_correction_messages(
    adapter: Any,
    benchmark_name: str,
    sample: Dict[str, Any],
    response: Any,
    initial_extracted_res: Any,
    initial_score: float,
) -> List[Dict[str, Any]]:
    input_data = "\n\n".join([
        xml_block("question", sample.get("question"), escape=True),
        xml_block("gold_truth", sample.get("answer"), escape=True),
        xml_block("gold_answer_format", sample.get("answer_format"), escape=True),
        xml_block("model_response", response, escape=True),
        xml_block("initial_formatted_extraction", _format_initial_extraction(initial_extracted_res), escape=True),
        xml_block("scoring_code", _score_function_source(adapter, sample.get("answer_format")), cdata=True),
        xml_block("initial_score", initial_score, inline=True),
    ])
    prompt = xml_block("correction_prompt", "\n\n".join([
        xml_block(
            "role",
            "You are auditing a benchmark answer extraction after code-based scoring.",
            escape=True,
            indent=2,
        ),
        xml_block(
            "objective",
            "Recover the benchmark answer when model_response supports it but the initial extraction or answer format caused a false negative.",
            escape=True,
            indent=2,
        ),
        xml_block(
            "correction_policy",
            "Use gold_truth as a reference target for what to look for in model_response, not as an independent source of facts. Actively fix representation errors when model_response contains or directly entails the answer but the initial extraction loses score because of formatting. Allowed corrections include units, scale words, punctuation, hyphenation, capitalization, numeric spelling, percentage-point wording, list serialization, answer-format labels, aliases, paraphrases, and adding missing list items that are supported by model_response. Prefer the benchmark-friendly surface form when it is supported by model_response. For list answers, output the supported items at the same granularity and serialization style as gold_truth when model_response clearly mentions them. Do not copy gold_truth into the answer unless model_response supports it.",
            escape=True,
            indent=2,
        ),
        xml_block("input_data", input_data),
        xml_block(
            "decision_rule",
            "Return a changed extraction when model_response supports gold_truth in a direct, paraphrased, normalized numeric/unit, alias, or list-equivalent form. If model_response is substantively wrong or does not contain enough evidence, return initial_formatted_extraction unchanged.",
            escape=True,
            indent=2,
        ),
        xml_block(
            "thinking_policy",
            "First output one <think>...</think> block. Compare model_response, gold_truth, initial_formatted_extraction, initial_score, and scoring_code. State whether the correction is supported by model_response or whether the initial extraction must be preserved.",
            escape=True,
            indent=2,
        ),
        xml_block("output_schema", _correction_output_contract(benchmark_name), cdata=True, indent=2),
        xml_block(
            "self_check",
            "Before returning, verify that corrected_extraction is parseable, uses the benchmark-specific extraction format, does not introduce unsupported facts from gold_truth, and is unchanged when model_response does not support a correction.",
            escape=True,
            indent=2,
        ),
        _correction_few_shot_examples_xml(benchmark_name),
    ]))
    return [{"role": "user", "content": text_content_parts(prompt)}]


def _unwrap_corrected_extraction(text: Any) -> str:
    content = _strip_leading_think_block(text)
    extraction = _unwrap_xml_block(content, "corrected_extraction")
    if extraction is not None:
        return extraction
    return content


def _apply_correction_if_needed(
    adapter: Any,
    benchmark_name: str,
    sample: Dict[str, Any],
    cfg,
    client,
    response: Any,
    initial_extracted_res: Any,
    parse_extraction_result,
    log_prefix: str,
) -> None:
    initial_score = float(sample["score"])
    if initial_score >= 1.0:
        return
    if not bool(get_config_value(cfg, "benchmarks.correction_enabled", True)):
        return

    correction_model_name = get_config_value(cfg, "benchmarks.correction_model_name")
    if not correction_model_name:
        correction_model_name = require_config_value(cfg, "benchmarks.extractor_model_name")

    initial_pred = sample.get("pred")
    initial_pred_format = sample.get("pred_format")
    correction_start = perf_counter()
    messages = _build_correction_messages(
        adapter,
        benchmark_name,
        sample,
        response,
        initial_extracted_res,
        initial_score,
    )
    metadata: Dict[str, Any] = {
        "input_messages": to_plain_data(messages),
        "initial_pred": to_plain_data(initial_pred),
        "initial_pred_format": initial_pred_format,
        "initial_score": initial_score,
        "changed": False,
        "improved": False,
        "applied": False,
    }
    try:
        completion = call_llm_messages(
            client,
            correction_model_name,
            messages,
            max_tokens=8192,
            temperature=0.0,
            retries=3,
            logger=logger,
            log_prefix=log_prefix,
            failure_value=lambda exc: f"Failed: {exc}",
        )
        corrected_extracted_res = completion_content(completion)
        metadata.update(_completion_metadata(completion, correction_model_name))
        metadata["corrected_extracted_res"] = corrected_extracted_res
        corrected_pred, corrected_pred_format = parse_extraction_result(_unwrap_corrected_extraction(corrected_extracted_res))
        if corrected_pred is None:
            metadata["error"] = "Failed to parse corrected extraction"
            return
        corrected_score = adapter.score(sample["answer"], corrected_pred, sample["answer_format"])
        metadata["corrected_pred"] = to_plain_data(corrected_pred)
        metadata["corrected_pred_format"] = corrected_pred_format
        metadata["corrected_score"] = corrected_score
        metadata["changed"] = (
            str(to_plain_data(corrected_pred)) != str(to_plain_data(initial_pred))
            or str(corrected_pred_format) != str(initial_pred_format)
        )
        metadata["improved"] = corrected_score > initial_score
        if corrected_score < initial_score:
            return
        metadata["applied"] = True
    except Exception as exc:
        metadata["error"] = str(exc)
    finally:
        metadata["duration_seconds"] = round(perf_counter() - correction_start, 3)
        sample["correction_metadata"] = metadata


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


def _normalize_list_candidate(value: Any) -> list[Any]:
    parsed = _maybe_parse_literal(value)
    if isinstance(parsed, list):
        return parsed
    if not isinstance(parsed, str):
        return [parsed]
    text = parsed.strip()
    if not text:
        return [parsed]
    parts = [part.strip() for part in re.split(r"\s*,\s*", text) if part.strip()]
    return parts if len(parts) > 1 else [parsed]


def _normalize_list_item(value: Any) -> str:
    text = str(value).lower().strip()
    text = re.sub(r"\bpercentage\s+points?\b", "%", text)
    text = re.sub(r"\bpercent(age)?\b", "%", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"\1%", text)
    text = re.sub(r"\s+households?\b", "", text).strip()
    return MMLongBenchAdapter._clean_string(text)


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
            gt = _normalize_list_candidate(gt)
            pred = _normalize_list_candidate(pred)
            if len(gt) != len(pred):
                score = 0.0
            else:
                gt = sorted([_normalize_list_item(item) for item in gt])
                pred = sorted([_normalize_list_item(item) for item in pred])
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
        samples = _apply_sample_limit(load_mmlongbench_samples(input_path), cfg)
        logger.info("Loaded %s MMLongBench samples from %s", len(samples), input_path)
        return samples

    def sample_key(self, sample: Dict[str, Any]) -> Any:
        return (sample.get("doc_id"), sample.get("question"), sample.get("answer"), sample.get("answer_format"))

    def is_successful_result(self, sample: Dict[str, Any]) -> bool:
        return "score" in sample and sample.get("pred") != "Failed to extract"

    @staticmethod
    def parse_extraction_result(extracted_res: Any) -> tuple[str | None, str | None]:
        text = _unwrap_extraction_result(extracted_res)
        answer_match = re.search(r"Extracted answer:\s*(.*?)(?:\n+Answer format:|$)", text, flags=re.DOTALL)
        if not answer_match:
            return None, None
        format_match = re.search(r"Answer format:\s*(.*?)(?:\n|$)", text, flags=re.DOTALL)
        pred_format = _unwrap_inline_xml_value(format_match.group(1).strip(), ("answer_format",)) if format_match else None
        pred = _unwrap_inline_xml_value(answer_match.group(1).strip(), ("answer", "concise_answer"))
        return pred, pred_format

    def build_extraction_messages(
        self,
        sample: Dict[str, Any],
        response: Any,
    ) -> List[Dict[str, Any]]:
        prompt = _build_extraction_prompt("mmlongbench", sample.get("question"), response)
        return [{"role": "user", "content": text_content_parts(prompt)}]

    def process_sample(self, sample: Dict[str, Any], cfg, context_builder, client) -> Dict[str, Any] | None:
        sample = dict(sample)
        _reset_sample_fields(sample, COMMON_RESULT_FIELDS)
        self_answered = _process_self_answering_sample(sample, context_builder, "mmlongbench", client)
        if self_answered is not None:
            response = self_answered["generation_metadata"].get("response")
            extractor_model_name = require_config_value(cfg, "benchmarks.extractor_model_name")
        else:
            messages, qa_model_name, extractor_model_name = _prepare_messages(sample, cfg, context_builder, "mmlongbench", client)
            response = _generate_response(sample, messages, qa_model_name, client, "MMLongBench generation")
        if response is None:
            return None
        extraction_messages = self.build_extraction_messages(sample, response)
        pred, pred_format = _extract_prediction(
            sample,
            extraction_messages,
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
        _apply_correction_if_needed(
            self,
            "mmlongbench",
            sample,
            cfg,
            client,
            response,
            sample["extraction_metadata"]["extracted_res"],
            self.parse_extraction_result,
            "MMLongBench correction",
        )
        return _finalize_result_fields(sample)

    def build_metrics(self, samples: List[Dict[str, Any]], output_path: Path) -> Dict[str, Any]:
        completed = [sample for sample in samples if self.is_successful_result(sample)]
        final_completed = [_final_prediction_sample(sample) for sample in completed]
        overall_acc, overall_f1 = self.acc_and_f1(final_completed)
        metrics: Dict[str, Any] = {"overall_acc": overall_acc, "overall_f1": overall_f1}

        def group_metrics(group_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
            final_group_samples = [_final_prediction_sample(sample) for sample in group_samples]
            acc, f1 = self.acc_and_f1(final_group_samples)
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
        samples = _apply_sample_limit(load_longdocurl_samples(qa_file, image_prefix=image_prefix), cfg)
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
        prompt = _build_extraction_prompt("longdocurl", sample.get("question"), response)
        return [{"role": "user", "content": text_content_parts(prompt)}]

    def parse_extraction_result(self, extracted_res: Any) -> tuple[Any, str | None]:
        text = _unwrap_extraction_result(extracted_res)
        try:
            concise_answer = re.findall(r"<concise_answer>(.*?)</concise_answer>", text, re.DOTALL)[0]
        except Exception:
            return None, None
        format_match = re.search(r"<answer_format>(.*?)</answer_format>", text, re.DOTALL)
        pred_format = format_match.group(1).strip() if format_match else None
        return self.parse_concise_answer(concise_answer), pred_format

    def process_sample(self, sample: Dict[str, Any], cfg, context_builder, client) -> Dict[str, Any] | None:
        sample = dict(sample)
        _reset_sample_fields(sample, COMMON_RESULT_FIELDS)
        self_answered = _process_self_answering_sample(sample, context_builder, "longdocurl", client)
        if self_answered is not None:
            response = self_answered["generation_metadata"].get("response")
            extractor_model_name = require_config_value(cfg, "benchmarks.extractor_model_name")
        else:
            messages, qa_model_name, extractor_model_name = _prepare_messages(sample, cfg, context_builder, "longdocurl", client)
            response = _generate_response(sample, messages, qa_model_name, client, "LongDocURL generation")
        if response is None:
            return None
        extraction_messages = self.build_extraction_messages(sample, response)
        pred, pred_format = _extract_prediction(
            sample,
            extraction_messages,
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
        _apply_correction_if_needed(
            self,
            "longdocurl",
            sample,
            cfg,
            client,
            response,
            sample["extraction_metadata"]["extracted_res"],
            self.parse_extraction_result,
            "LongDocURL correction",
        )
        return _finalize_result_fields(sample)

    def build_metrics(self, samples: List[Dict[str, Any]], output_path: Path) -> Dict[str, Any]:
        completed = [sample for sample in samples if self.is_successful_result(sample)]
        final_completed = [_final_prediction_sample(sample) for sample in completed]
        avg_acc = self.accuracy(
            [sample["pred"] for sample in final_completed],
            [sample["answer"] for sample in final_completed],
            [sample["answer_format"] for sample in final_completed],
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
            fine_grained_samples = [dict(sample) for sample in final_completed]
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
