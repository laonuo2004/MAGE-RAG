import json
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

import benchmarks.adapters as adapters
from baselines.base import ContextMessages
from benchmarks.adapters import LongDocURLAdapter, MMLongBenchAdapter
from utils.llm_utils import text_content_parts


class StubContextBuilder:
    def build(self, benchmark_name, sample):
        return ContextMessages(
            [{"role": "user", "content": text_content_parts(f"{benchmark_name}:{sample['question']}")}],
            metadata={"context_builder": "stub", "sample_question": sample["question"]},
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

    def test_mmlongbench_process_sample_uses_shared_llm_call_and_fields(self):
        calls = []
        original_call_llm_messages = adapters.call_llm_messages
        try:
            def fake_call_llm_messages(*args, **kwargs):
                calls.append((args, kwargs))
                if len(calls) == 1:
                    return "raw answer"
                return "Extracted answer: answer\nAnswer format: String"

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
                "context_metadata": {"stale": True},
                "pred": "stale",
                "score": 0.0,
                "timing_total_seconds": 99.0,
            }

            result = MMLongBenchAdapter().process_sample(sample, cfg, StubContextBuilder(), object())
        finally:
            adapters.call_llm_messages = original_call_llm_messages

        self.assertEqual([call[0][1] for call in calls], ["qa-model", "extractor-model"])
        self.assertEqual(calls[1][0][2][0]["content"][0]["type"], "text")
        self.assertIn("Question: q", calls[1][0][2][0]["content"][0]["text"])
        self.assertIn("Analysis: raw answer", calls[1][0][2][0]["content"][0]["text"])
        self.assertEqual(result["response"], "raw answer")
        self.assertEqual(result["pred"], "answer")
        self.assertIn("score", result)
        self.assertIn("timing_prepare_seconds", result)
        self.assertIn("timing_generation_seconds", result)
        self.assertIn("timing_extraction_seconds", result)
        self.assertEqual(result["context_metadata"]["context_builder"], "stub")
        self.assertNotIn("stale", result["context_metadata"])
        self.assertNotIn("timing_total_seconds", result)

    def test_longdocurl_process_sample_uses_shared_llm_call_and_fields(self):
        calls = []
        original_call_llm_messages = adapters.call_llm_messages
        try:
            def fake_call_llm_messages(*args, **kwargs):
                calls.append((args, kwargs))
                if len(calls) == 1:
                    return "analysis"
                return "<concise_answer>'answer'</concise_answer>"

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
                "context_metadata": {"stale": True},
                "pred": "stale",
                "score": 0.0,
                "timing_total_seconds": 99.0,
            }

            result = LongDocURLAdapter().process_sample(sample, cfg, StubContextBuilder(), object())
        finally:
            adapters.call_llm_messages = original_call_llm_messages

        self.assertEqual([call[0][1] for call in calls], ["qa-model", "extractor-model"])
        self.assertEqual(calls[1][0][2][0]["content"][0]["type"], "text")
        self.assertIn("Analysis: analysis", calls[1][0][2][0]["content"][0]["text"])
        self.assertEqual(result["response"], "analysis")
        self.assertEqual(result["pred"], "answer")
        self.assertIn("score", result)
        self.assertIn("timing_prepare_seconds", result)
        self.assertIn("timing_generation_seconds", result)
        self.assertIn("timing_extraction_seconds", result)
        self.assertEqual(result["context_metadata"]["sample_question"], "q")
        self.assertNotIn("stale", result["context_metadata"])
        self.assertNotIn("timing_total_seconds", result)


if __name__ == "__main__":
    unittest.main()
