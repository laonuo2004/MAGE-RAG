import json
import os
import tempfile
import unittest

from omegaconf import OmegaConf

import benchmarks.mmlongbench.run_api as run_api
from baselines.base import ContextMessages
from benchmarks.results import RESULTS_ROOT
from benchmarks.mmlongbench.run_api import (
    append_result,
    build_default_results_file,
    compact_results_file,
    handle_sample_result,
    read_jsonl_results,
    request_llm,
)
from utils.config_utils import require_config_value


class MMLongBenchResultsTests(unittest.TestCase):
    def test_default_results_file_uses_jsonl(self):
        cfg = OmegaConf.create({
            "baselines": {"name": "image"},
            "benchmarks": {
                "name": "mmlongbench",
                "results_dir": "/tmp/mmlongbench-results",
                "qa_model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
            },
        })
        benchmark_cfg = require_config_value(cfg, "benchmarks")

        self.assertEqual(
            build_default_results_file(cfg, benchmark_cfg),
            str(RESULTS_ROOT / "mmlongbench/image/res_Qwen_Qwen2_5_VL_7B_Instruct.jsonl"),
        )

    def test_default_results_file_includes_m3docrag_top_k(self):
        cfg = OmegaConf.create({
            "baselines": {"name": "m3docrag", "top_k": 3},
            "benchmarks": {
                "name": "mmlongbench",
                "results_dir": "/tmp/mmlongbench-results",
                "qa_model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
            },
        })
        benchmark_cfg = require_config_value(cfg, "benchmarks")

        self.assertEqual(
            build_default_results_file(cfg, benchmark_cfg),
            str(RESULTS_ROOT / "mmlongbench/m3docrag/res_top_k_3_Qwen_Qwen2_5_VL_7B_Instruct.jsonl"),
        )

    def test_append_and_read_jsonl_results(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "results.jsonl")
            append_result({"doc_id": "d1", "score": 1.0}, output_path)
            append_result({"doc_id": "d2", "score": 0.0}, output_path)
            with open(output_path, "a", encoding="utf-8") as f:
                f.write("{bad json}\n")

            with self.assertLogs("mmlongbench.run_api", level="WARNING"):
                samples = read_jsonl_results(output_path)

        self.assertEqual(samples, [{"doc_id": "d1", "score": 1.0}, {"doc_id": "d2", "score": 0.0}])

    def test_append_writes_one_json_object_per_line(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "results.jsonl")
            append_result({"answer": ["A", "B"]}, output_path)

            with open(output_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0]), {"answer": ["A", "B"]})

    def test_read_results_ignores_failed_extract_records(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "results.jsonl")
            append_result({"doc_id": "d1", "pred": "Failed to extract", "score": 0.0}, output_path)
            append_result({"doc_id": "d2", "pred": "ok", "score": 0.0}, output_path)

            samples = read_jsonl_results(output_path)

        self.assertEqual(samples, [{"doc_id": "d2", "pred": "ok", "score": 0.0}])

    def test_compact_results_removes_failed_extract_records(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "results.jsonl")
            append_result({"doc_id": "d1", "question": "q1", "answer": "a", "answer_format": "Str", "pred": "Failed to extract", "score": 0.0}, output_path)
            append_result({"doc_id": "d2", "question": "q2", "answer": "a", "answer_format": "Str", "pred": "ok", "score": 0.0}, output_path)

            compact_results_file(output_path)

            with open(output_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["doc_id"], "d2")

    def test_handle_result_does_not_append_failed_sample(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "results.jsonl")
            samples = [{"doc_id": "d1", "question": "q?", "answer": "a", "answer_format": "Str"}]

            with self.assertLogs("mmlongbench.run_api", level="WARNING"):
                handle_sample_result(None, samples, 0, output_path)

            written = read_jsonl_results(output_path)

        self.assertEqual(written, [])
        self.assertEqual(samples, [{"doc_id": "d1", "question": "q?", "answer": "a", "answer_format": "Str"}])

    def test_handle_result_appends_success_sample(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "results.jsonl")
            samples = [{"doc_id": "d1", "question": "q?", "answer": "a", "answer_format": "Str"}]
            sample = {
                "doc_id": "d1",
                "question": "q?",
                "answer": "a",
                "answer_format": "Str",
                "response": "analysis",
                "pred": "a",
                "score": 1.0,
            }

            handle_sample_result(sample, samples, 0, output_path)

            written = read_jsonl_results(output_path)

        self.assertEqual(written, [sample])
        self.assertEqual(samples, [sample])

    def test_request_llm_retries_none_content(self):
        class Message:
            def __init__(self, content):
                self.content = content

        class Choice:
            def __init__(self, content):
                self.message = Message(content)

        class Response:
            def __init__(self, content):
                self.choices = [Choice(content)]

        class Completions:
            def __init__(self):
                self.contents = [None, "ok"]
                self.calls = 0

            def create(self, **kwargs):
                content = self.contents[self.calls]
                self.calls += 1
                return Response(content)

        class Chat:
            def __init__(self):
                self.completions = Completions()

        class Client:
            def __init__(self):
                self.chat = Chat()

        client = Client()

        with self.assertLogs("mmlongbench.run_api", level="WARNING"):
            self.assertEqual(request_llm([], "model", client), "ok")
        self.assertEqual(client.chat.completions.calls, 2)

    def test_process_one_sample_persists_context_metadata(self):
        class Builder:
            def build(self, benchmark_name, sample):
                return ContextMessages(
                    [{"role": "user", "content": "prompt"}],
                    metadata={"context_builder": "mock", "retrieved_pages": [{"page_index": 0}]},
                )

        cfg = OmegaConf.create({
            "baselines": {"name": "mock"},
            "benchmarks": {
                "qa_model_name": "qa-model",
                "extractor_model_name": "extractor-model",
            },
            "litellm": {
                "api_key": "key",
                "base_url": "http://localhost",
            },
        })
        sample = {
            "doc_id": "d1",
            "question": "q?",
            "answer": "a",
            "answer_format": "Str",
        }

        original_build_context_builder = run_api.build_context_builder
        original_build_openai_client = run_api.build_openai_client
        original_request_llm = run_api.request_llm
        original_extract_answer = run_api.extract_answer
        original_eval_score = run_api.eval_score
        try:
            run_api.build_context_builder = lambda cfg: Builder()
            run_api.build_openai_client = lambda cfg: object()
            run_api.request_llm = lambda messages, model_name, client: "analysis"
            run_api.extract_answer = lambda question, response, prompt, model_name, client: (
                "Extracted answer: a\nAnswer format: Str"
            )
            run_api.eval_score = lambda answer, pred, answer_format: 1.0

            result = run_api.process_one_sample(sample, cfg, "extractor prompt")
        finally:
            run_api.build_context_builder = original_build_context_builder
            run_api.build_openai_client = original_build_openai_client
            run_api.request_llm = original_request_llm
            run_api.extract_answer = original_extract_answer
            run_api.eval_score = original_eval_score

        self.assertEqual(
            result["context_metadata"],
            {"context_builder": "mock", "retrieved_pages": [{"page_index": 0}]},
        )


if __name__ == "__main__":
    unittest.main()
