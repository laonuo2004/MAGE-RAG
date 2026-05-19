import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

import benchmarks.adapters as adapters
from baselines.base import ContextMessages
from benchmarks.adapters import LongDocURLAdapter, MMLongBenchAdapter
from utils.llm_utils import text_content_parts


class StubContextBuilder:
    def build(self, benchmark_name, sample, **kwargs):
        self.last_kwargs = kwargs
        return ContextMessages(
            [{"role": "user", "content": text_content_parts(f"{benchmark_name}:{sample['question']}")}],
            metadata={"context_builder": "stub", "sample_question": sample["question"]},
        )


def completion(content, *, model="served-model", prompt_tokens=10, completion_tokens=2):
    return SimpleNamespace(
        id="cmpl-test",
        model=model,
        created=123,
        system_fingerprint="fp-test",
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


class BenchmarkAdapterTests(unittest.TestCase):
    def test_mmlongbench_loads_json_array_and_uses_stable_key(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "samples.json"
            sample = {"doc_id": "d1", "question": "q", "answer": "a", "answer_format": "Str"}
            input_path.write_text(json.dumps([sample]), encoding="utf-8")
            cfg = OmegaConf.create({"benchmarks": {"input_path": str(input_path)}})

            adapter = MMLongBenchAdapter()
            samples = adapter.load_samples(cfg)

        self.assertEqual(samples, [sample])
        self.assertEqual(adapter.sample_key(sample), ("d1", "q", "a", "Str"))

    def test_longdocurl_loads_jsonl_adds_question_id_and_image_prefix(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            qa_file = Path(tmp_dir) / "qa.jsonl"
            qa_file.write_text(
                json.dumps({
                    "question": "q",
                    "answer": "a",
                    "answer_format": "Str",
                    "images": ["/old/doc/page_1.png"],
                }) + "\n",
                encoding="utf-8",
            )
            cfg = OmegaConf.create({
                "benchmarks": {
                    "qa_file": str(qa_file),
                    "image_prefix": str(Path(tmp_dir) / "images"),
                }
            })

            adapter = LongDocURLAdapter()
            samples = adapter.load_samples(cfg)

        self.assertEqual(samples[0]["question_id"], 0)
        self.assertEqual(adapter.sample_key(samples[0]), 0)
        self.assertTrue(samples[0]["images"][0].endswith("images/doc/page_1.png"))

    def test_score_success_field_is_unified(self):
        self.assertTrue(MMLongBenchAdapter().is_successful_result({"pred": "ok", "score": 1.0}))
        self.assertTrue(LongDocURLAdapter().is_successful_result({"pred": "ok", "score": 1.0}))
        self.assertFalse(LongDocURLAdapter().is_successful_result({"pred": "ok", "score_v3": 1.0}))

    def test_mmlongbench_metrics_include_fine_grained_json_breakdowns(self):
        samples = [
            {
                "answer": "alpha",
                "pred": "alpha",
                "score": 1.0,
                "evidence_pages": "[1]",
                "evidence_sources": "['Figure', 'Table']",
                "doc_type": "Guidebook",
                "answer_format": "Str",
            },
            {
                "answer": "beta",
                "pred": "Not answerable",
                "score": 0.0,
                "evidence_pages": "[1, 2]",
                "evidence_sources": "['Figure']",
                "doc_type": "Guidebook",
                "answer_format": "Str",
            },
            {
                "answer": "Not answerable",
                "pred": "Not answerable",
                "score": 1.0,
                "evidence_pages": "[]",
                "evidence_sources": "[]",
                "doc_type": "Financial report",
                "answer_format": "None",
            },
            {"answer": "gamma", "pred": "Failed to extract"},
        ]

        metrics = MMLongBenchAdapter().build_metrics(samples, Path("result.jsonl"))

        self.assertEqual(metrics["breakdowns"]["single_page"]["count"], 1)
        self.assertEqual(metrics["breakdowns"]["cross_page"]["count"], 1)
        self.assertEqual(metrics["breakdowns"]["unanswerable"]["count"], 1)
        self.assertEqual(metrics["evidence_source_breakdowns"]["Figure"]["count"], 2)
        self.assertEqual(metrics["evidence_source_breakdowns"]["Figure"]["acc"], 0.5)
        self.assertEqual(metrics["evidence_source_breakdowns"]["Table"]["count"], 1)
        self.assertEqual(metrics["document_type_breakdowns"]["Guidebook"]["count"], 2)
        self.assertEqual(metrics["document_type_breakdowns"]["Guidebook"]["acc"], 0.5)
        self.assertEqual(metrics["answer_format_breakdowns"]["Str"]["count"], 2)
        self.assertNotIn("text_metrics_file", metrics)

    def test_mmlongbench_process_sample_uses_shared_llm_call_and_fields(self):
        calls = []
        original_call_llm_messages = adapters.call_llm_messages
        try:
            def fake_call_llm_messages(*args, **kwargs):
                calls.append((args, kwargs))
                if len(calls) == 1:
                    return completion("raw answer", model="qa-served", prompt_tokens=11, completion_tokens=3)
                return completion(
                    "Extracted answer: answer\nAnswer format: String",
                    model="extractor-served",
                    prompt_tokens=7,
                    completion_tokens=5,
                )

            adapters.call_llm_messages = fake_call_llm_messages
            cfg = OmegaConf.create({
                "benchmarks": {
                    "qa_model_name": "qa-model",
                    "extractor_model_name": "extractor-model",
                }
            })
            sample = {
                "doc_id": "d1",
                "question": "q",
                "answer": "answer",
                "answer_format": "String",
                "prepare_metadata": {"stale": True},
                "pred": "stale",
                "score": 0.0,
            }

            stub_builder = StubContextBuilder()
            client = object()
            result = MMLongBenchAdapter().process_sample(sample, cfg, stub_builder, client)
        finally:
            adapters.call_llm_messages = original_call_llm_messages

        self.assertIs(stub_builder.last_kwargs["client"], client)
        self.assertEqual([call[0][1] for call in calls], ["qa-model", "extractor-model"])
        self.assertEqual(calls[1][0][2][0]["content"][0]["type"], "text")
        self.assertIn("Question: q", calls[1][0][2][0]["content"][0]["text"])
        self.assertIn("Analysis: raw answer", calls[1][0][2][0]["content"][0]["text"])
        self.assertEqual(result["generation_metadata"]["response"], "raw answer")
        self.assertEqual(result["pred"], "answer")
        self.assertEqual(result["pred_format"], "String")
        self.assertEqual(result["prepare_metadata"]["context_builder"], "stub")
        self.assertEqual(result["generation_metadata"]["model"], "qa-served")
        self.assertEqual(result["generation_metadata"]["finish_reason"], "stop")
        self.assertEqual(result["generation_metadata"]["usage"]["total_tokens"], 14)
        self.assertEqual(result["extraction_metadata"]["model"], "extractor-served")
        self.assertEqual(result["extraction_metadata"]["extracted_res"], "Extracted answer: answer\nAnswer format: String")
        self.assertEqual(result["extraction_metadata"]["usage"]["total_tokens"], 12)
        self.assertIn("score", result)
        self.assertIn("duration_seconds", result["prepare_metadata"])
        self.assertIn("duration_seconds", result["generation_metadata"])
        self.assertIn("duration_seconds", result["extraction_metadata"])
        self.assertNotIn("stale", result["prepare_metadata"])

    def test_longdocurl_process_sample_uses_shared_llm_call_and_fields(self):
        calls = []
        original_call_llm_messages = adapters.call_llm_messages
        try:
            def fake_call_llm_messages(*args, **kwargs):
                calls.append((args, kwargs))
                if len(calls) == 1:
                    return completion("analysis", model="qa-served", prompt_tokens=13, completion_tokens=4)
                return completion(
                    "<concise_answer>'answer'</concise_answer><answer_format>String</answer_format>",
                    model="extractor-served",
                    prompt_tokens=8,
                    completion_tokens=6,
                )

            adapters.call_llm_messages = fake_call_llm_messages
            cfg = OmegaConf.create({
                "benchmarks": {
                    "qa_model_name": "qa-model",
                    "extractor_model_name": "extractor-model",
                }
            })
            sample = {
                "question_id": 1,
                "question": "q",
                "answer": "answer",
                "answer_format": "String",
                "prepare_metadata": {"stale": True},
                "pred": "stale",
                "score": 0.0,
            }

            result = LongDocURLAdapter().process_sample(sample, cfg, StubContextBuilder(), object())
        finally:
            adapters.call_llm_messages = original_call_llm_messages

        self.assertEqual([call[0][1] for call in calls], ["qa-model", "extractor-model"])
        self.assertEqual(calls[1][0][2][0]["content"][0]["type"], "text")
        self.assertIn("Analysis: analysis", calls[1][0][2][0]["content"][0]["text"])
        self.assertEqual(result["generation_metadata"]["response"], "analysis")
        self.assertEqual(result["pred"], "answer")
        self.assertEqual(result["pred_format"], "String")
        self.assertEqual(result["prepare_metadata"]["sample_question"], "q")
        self.assertEqual(result["generation_metadata"]["model"], "qa-served")
        self.assertEqual(result["generation_metadata"]["usage"]["total_tokens"], 17)
        self.assertEqual(result["extraction_metadata"]["model"], "extractor-served")
        self.assertEqual(result["extraction_metadata"]["finish_reason"], "stop")
        self.assertIn("score", result)
        self.assertIn("duration_seconds", result["prepare_metadata"])
        self.assertIn("duration_seconds", result["generation_metadata"])
        self.assertIn("duration_seconds", result["extraction_metadata"])
        self.assertNotIn("stale", result["prepare_metadata"])


if __name__ == "__main__":
    unittest.main()
