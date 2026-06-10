import json
import tempfile
import time
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


class PartiallyFailingAdapter(DummyAdapter):
    def load_samples(self, cfg):
        return [{"id": "a"}, {"id": "b", "always_fail": True}]

    def process_sample(self, sample, cfg, context_builder, client):
        if sample.get("always_fail"):
            return None
        return super().process_sample(sample, cfg, context_builder, client)


class WriteFailureAdapter(DummyAdapter):
    def process_sample(self, sample, cfg, context_builder, client):
        if sample["id"] == "bad":
            return {"id": "bad", "pred": "ok", "score": 1.0, "not_json": {"x"}}
        time.sleep(0.05)
        return {"id": sample["id"], "pred": "ok", "score": 1.0}


class RunnerResultsTests(unittest.TestCase):
    def test_all_baseline_configs_declare_result_name_params(self):
        for path in sorted(Path("configs/baselines").glob("*.yaml")):
            with self.subTest(path=str(path)):
                cfg = OmegaConf.load(path)
                self.assertIn("result_name_params", cfg)

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

    def test_result_name_params_select_nested_config_values(self):
        cfg = OmegaConf.create({
            "benchmarks": {"name": "mmlongbench", "qa_model_name": "model"},
            "baselines": {
                "name": "magerag",
                "params": {"top_k": 3},
                "controller": {"watchdog_iterations": 6},
                "result_name_params": [
                    "params.top_k",
                    "controller.watchdog_iterations",
                ],
            },
        })

        self.assertEqual(
            str(build_results_file(cfg)),
            str(RESULTS_ROOT / "mmlongbench/magerag/res_top_k_3_watchdog_iterations_6_model.jsonl"),
        )

    def test_empty_result_name_params_uses_model_only_filename(self):
        cfg = OmegaConf.create({
            "benchmarks": {"name": "mmlongbench", "qa_model_name": "model"},
            "baselines": {
                "name": "image",
                "result_name_params": [],
            },
        })

        self.assertEqual(
            str(build_results_file(cfg)),
            str(RESULTS_ROOT / "mmlongbench/image/res_model.jsonl"),
        )

    def test_missing_result_name_param_path_raises(self):
        cfg = OmegaConf.create({
            "benchmarks": {"name": "mmlongbench", "qa_model_name": "model"},
            "baselines": {
                "name": "magerag",
                "params": {"top_k": 3},
                "result_name_params": ["controller.watchdog_iterations"],
            },
        })

        with self.assertRaisesRegex(ValueError, "Missing result_name_params config value"):
            build_results_file(cfg)

    def test_result_name_params_preserve_null_values(self):
        cfg = OmegaConf.create({
            "benchmarks": {"name": "mmlongbench", "qa_model_name": "model"},
            "baselines": {
                "name": "bgem3",
                "params": {"max_cross_pages": None},
                "result_name_params": ["params.max_cross_pages"],
            },
        })

        self.assertEqual(
            str(build_results_file(cfg)),
            str(RESULTS_ROOT / "mmlongbench/bgem3/res_max_cross_pages_None_model.jsonl"),
        )

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

    def test_compact_results_file_normalizes_correction_field_order(self):
        adapter = DummyAdapter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.jsonl"
            append_jsonl({
                "id": "a",
                "pred": "fixed",
                "pred_format": "String",
                "score": 1.0,
                "corrected_pred": "fixed",
                "corrected_format": "String",
                "corrected_score": 1.0,
                "prepare_metadata": {},
                "generation_metadata": {},
                "extraction_metadata": {},
                "correction_metadata": {
                    "initial_pred": "raw",
                    "initial_pred_format": "String",
                    "initial_score": 0.0,
                    "corrected_pred": "fixed",
                    "corrected_pred_format": "String",
                    "corrected_score": 1.0,
                    "applied": True,
                },
            }, output_path)
            output_path.write_text(output_path.read_text(encoding="utf-8") + "{bad json}\n", encoding="utf-8")

            compact_results_file(adapter, output_path)
            row = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(row["pred"], "raw")
        self.assertEqual(row["score"], 0.0)
        self.assertEqual(row["corrected_pred"], "fixed")
        self.assertEqual(row["corrected_score"], 1.0)
        keys = list(row)
        self.assertEqual(
            keys[keys.index("prepare_metadata"):keys.index("corrected_score") + 1],
            [
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
            ],
        )

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

    def test_run_pending_writes_enriched_result_envelope(self):
        adapter = DummyAdapter()
        cfg = OmegaConf.create({
            "benchmarks": {
                "name": "longdocurl",
                "process_mode": "serial",
                "workers": 1,
                "qa_model_name": "qa-model",
            },
            "baselines": {
                "name": "dummy",
                "params": {"top_k": 1},
            },
        })
        samples = [{"id": "a"}]
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.jsonl"
            run_pending(adapter, cfg, samples, output_path, object(), object())
            written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(written[0]["benchmark"], "longdocurl")
        self.assertEqual(written[0]["baseline"], "dummy")
        self.assertEqual(written[0]["run_config"]["baseline"]["params"]["top_k"], 1)
        self.assertEqual(samples[0]["benchmark"], "longdocurl")

    def test_parallel_write_failure_does_not_stop_later_results(self):
        adapter = WriteFailureAdapter()
        cfg = OmegaConf.create({"benchmarks": {"process_mode": "parallel", "workers": 2}})
        samples = [{"id": "bad"}, {"id": "good"}]
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.jsonl"
            run_pending(adapter, cfg, samples, output_path, object(), object())
            written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(written, [{"id": "good", "pred": "ok", "score": 1.0}])
        self.assertNotIn("score", samples[0])
        self.assertEqual(samples[1]["score"], 1.0)

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

    def test_run_benchmark_writes_partial_metrics_when_samples_fail(self):
        adapter = PartiallyFailingAdapter()
        original_build_context_builder = runner.build_context_builder
        original_build_openai_client = runner.build_openai_client
        try:
            runner.build_context_builder = lambda cfg: object()
            runner.build_openai_client = lambda cfg: object()
            with tempfile.TemporaryDirectory() as tmp_dir:
                output_path = Path(tmp_dir) / "results.jsonl"
                cfg = OmegaConf.create({
                    "benchmarks": {
                        "process_mode": "serial",
                        "workers": 1,
                        "results_file": str(output_path),
                    },
                    "baselines": {"name": "dummy"},
                })
                result = run_benchmark_with_adapter(cfg, adapter)
                metrics_path = output_path.with_suffix(".metrics.json")
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        finally:
            runner.build_context_builder = original_build_context_builder
            runner.build_openai_client = original_build_openai_client

        self.assertEqual(result["metrics_file"], str(metrics_path))
        self.assertEqual(metrics["sample_count"], 2)
        self.assertEqual(metrics["completed_count"], 1)
        self.assertEqual(metrics["failed_count"], 1)


if __name__ == "__main__":
    unittest.main()
