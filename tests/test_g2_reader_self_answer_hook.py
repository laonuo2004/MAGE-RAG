from types import SimpleNamespace

from omegaconf import OmegaConf

import benchmarks.adapters as adapters
from benchmarks.adapters import LongDocURLAdapter, MMLongBenchAdapter


class SelfAnsweringBuilder:
    def __init__(self):
        self.calls = []

    def run_sample(self, benchmark_name, sample, client=None):
        self.calls.append((benchmark_name, sample, client))
        return {
            "response": "G2 says the answer is A and B.",
            "pred": "wrong raw pred should not be scored directly",
            "pred_format": None,
            "metadata": {"context_builder": "g2-reader", "sample_key": sample.get("question_id")},
            "usage": {"total_tokens": 3},
        }


def completion(content, *, model="extractor-served", prompt_tokens=7, completion_tokens=5):
    return SimpleNamespace(
        id="cmpl-test",
        model=model,
        created=123,
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=content),
            )
        ],
    )


def test_longdocurl_self_answering_response_flows_through_extractor():
    calls = []
    original_call_llm_messages = adapters.call_llm_messages
    try:
        def fake_call_llm_messages(*args, **kwargs):
            calls.append((args, kwargs))
            return completion("Extracted answer: <concise_answer>['A', 'B']</concise_answer>\nAnswer format: <answer_format>List</answer_format>")

        adapters.call_llm_messages = fake_call_llm_messages
        cfg = OmegaConf.create({
            "benchmarks": {
                "qa_model_name": "unused",
                "extractor_model_name": "extractor-model",
                "correction_enabled": True,
            }
        })
        sample = {
            "question_id": "q1",
            "question": "q",
            "answer": ["A", "B"],
            "answer_format": "List",
        }
        builder = SelfAnsweringBuilder()
        client = object()

        result = LongDocURLAdapter().process_sample(sample, cfg, builder, client)
    finally:
        adapters.call_llm_messages = original_call_llm_messages

    assert builder.calls[0][0] == "longdocurl"
    assert builder.calls[0][2] is client
    assert len(calls) == 1
    assert calls[0][0][1] == "extractor-model"
    assert "G2 says the answer is A and B." in calls[0][0][2][0]["content"][0]["text"]
    assert result["pred"] == ["A", "B"]
    assert result["pred_format"] == "List"
    assert result["score"] == 1.0
    assert result["generation_metadata"]["response"] == "G2 says the answer is A and B."
    assert result["generation_metadata"]["usage"] == {"total_tokens": 3}
    assert result["prepare_metadata"]["context_builder"] == "g2-reader"
    assert result["extraction_metadata"]["extracted_res"].startswith("Extracted answer:")


def test_mmlongbench_self_answering_response_flows_through_extractor():
    calls = []
    original_call_llm_messages = adapters.call_llm_messages
    try:
        def fake_call_llm_messages(*args, **kwargs):
            calls.append((args, kwargs))
            return completion("Extracted answer: ['A', 'B']\nAnswer format: List")

        adapters.call_llm_messages = fake_call_llm_messages
        cfg = OmegaConf.create({
            "benchmarks": {
                "qa_model_name": "unused",
                "extractor_model_name": "extractor-model",
                "correction_enabled": True,
            }
        })
        sample = {
            "doc_id": "doc.pdf",
            "question_id": "q1",
            "question": "q",
            "answer": "['A', 'B']",
            "answer_format": "List",
        }
        builder = SelfAnsweringBuilder()

        result = MMLongBenchAdapter().process_sample(sample, cfg, builder, object())
    finally:
        adapters.call_llm_messages = original_call_llm_messages

    assert builder.calls[0][0] == "mmlongbench"
    assert len(calls) == 1
    assert calls[0][0][1] == "extractor-model"
    assert result["pred"] == "['A', 'B']"
    assert result["pred_format"] == "List"
    assert result["score"] == 1.0
    assert result["prepare_metadata"]["context_builder"] == "g2-reader"
    assert result["extraction_metadata"]["extracted_res"] == "Extracted answer: [\'A\', \'B\']\nAnswer format: List"
