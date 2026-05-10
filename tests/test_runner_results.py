import json
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

import benchmarks.runner as runner
from benchmarks.runner import compact_results_file, merge_existing_samples, run_benchmark_with_adapter, run_pending
from benchmarks.utils.results_utils import RESULTS_ROOT, append_jsonl, build_results_file


class DummyAdapter:
    name = "dummy"

    def sample_key(self, sample):
        return sample["id"]

    def is_successful_result(self, sample):
        return "score" in sample and sample.get("pred") != "Failed"

    def load_samples(self, cfg):
        return [{"id": "a"}, {"id": "b"}]

    def process_sample(self, sample, cfg, context_builder, client):
        sample["context_builder_id"] = id(context_builder)
        sample["client_id"] = id(client)
        if sample.get("fail_once") and not sample.get("attempted"):
            sample["attempted"] = True
            return None
        result = dict(sample)
        result["pred"] = "ok"
        result["score"] = 1.0
        return result

    def build_metrics(self, samples, output_path):
        return {}


class RunnerResultsTests(unittest.TestCase):
    def test_stable_result_paths_partition_by_baseline_dir(self):
        cfg = OmegaConf.create({
            "benchmarks": {"name": "mmlongbench", "qa_model_name": "Qwen3-VL-8B-Instruct"},
            "baselines": {
                "name": "bm25",
                "params": {"top_k": 3, "chunk_size": 150, "chunk_overlap": 0},
            },
        })

        self.assertEqual(
            str(build_results_file(cfg)),
            str(RESULTS_ROOT / "mmlongbench/bm25/res_top_k_3_chunk_size_150_chunk_overlap_0_Qwen3_VL_8B_Instruct.jsonl"),
        )

    def test_parameter_changes_produce_different_filenames(self):
        cfg_a = OmegaConf.create({
            "benchmarks": {"name": "mmlongbench", "qa_model_name": "model"},
            "baselines": {
                "name": "bm25",
                "params": {"top_k": 3, "chunk_size": 150, "chunk_overlap": 0},
            },
        })
        cfg_b = OmegaConf.create({
            "benchmarks": {"name": "mmlongbench", "qa_model_name": "model"},
            "baselines": {
                "name": "bm25",
                "params": {"top_k": 5, "chunk_size": 150, "chunk_overlap": 0},
            },
        })

        self.assertNotEqual(build_results_file(cfg_a), build_results_file(cfg_b))

    def test_no_params_baseline_uses_model_only_filename(self):
        cfg = OmegaConf.create({
            "benchmarks": {"name": "mmlongbench", "qa_model_name": "model"},
            "baselines": {"name": "ocr"},
        })

        self.assertEqual(
            str(build_results_file(cfg)),
            str(RESULTS_ROOT / "mmlongbench/ocr/res_model.jsonl"),
        )

    def test_merge_existing_successes_and_compact_jsonl(self):
        adapter = DummyAdapter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.jsonl"
            append_jsonl({"id": "a", "pred": "ok", "score": 1.0}, output_path)
            append_jsonl({"id": "a", "pred": "ok2", "score": 0.5}, output_path)
            append_jsonl({"id": "b", "pred": "Failed"}, output_path)
            output_path.write_text(output_path.read_text(encoding="utf-8") + "{bad json}\n", encoding="utf-8")

            compact_results_file(adapter, output_path)
            merged = merge_existing_samples(adapter, [{"id": "a"}, {"id": "b"}], output_path)

            lines = output_path.read_text(encoding="utf-8").strip().splitlines()

        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["pred"], "ok2")
        self.assertEqual(merged[0]["score"], 0.5)
        self.assertNotIn("score", merged[1])

    def test_failed_samples_retry_on_next_round(self):
        adapter = DummyAdapter()
        cfg = OmegaConf.create({"benchmarks": {"process_mode": "serial", "workers": 1}})
        samples = [{"id": "a", "fail_once": True}, {"id": "b"}]
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.jsonl"
            context_builder = object()
            client = object()
            run_pending(adapter, cfg, samples, output_path, context_builder, client)
            run_pending(adapter, cfg, samples, output_path, context_builder, client)
            written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual([sample["id"] for sample in written], ["b", "a"])

    def test_run_benchmark_reuses_single_context_builder_and_client(self):
        adapter = DummyAdapter()
        context_builder = object()
        client = object()
        original_build_context_builder = runner.build_context_builder
        original_build_openai_client = runner.build_openai_client
        try:
            runner.build_context_builder = lambda cfg: context_builder
            runner.build_openai_client = lambda cfg: client
            with tempfile.TemporaryDirectory() as tmp_dir:
                cfg = OmegaConf.create({
                    "benchmarks": {
                        "process_mode": "serial",
                        "workers": 1,
                        "results_file": str(Path(tmp_dir) / "results.jsonl"),
                    },
                    "baselines": {"name": "dummy"},
                })
                result = run_benchmark_with_adapter(cfg, adapter)
        finally:
            runner.build_context_builder = original_build_context_builder
            runner.build_openai_client = original_build_openai_client

        samples = result["samples"]
        self.assertEqual({sample["context_builder_id"] for sample in samples}, {id(context_builder)})
        self.assertEqual({sample["client_id"] for sample in samples}, {id(client)})


if __name__ == "__main__":
    unittest.main()
