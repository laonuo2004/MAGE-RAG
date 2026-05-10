import json
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from benchmarks.report import generate_reports
from benchmarks.results import RESULTS_ROOT, append_jsonl, build_results_file, sidecar_paths, write_json
from benchmarks.runner import compact_results_file, merge_existing_samples, run_pending


class DummyAdapter:
    name = "dummy"

    def sample_key(self, sample):
        return sample["id"]

    def is_successful_result(self, sample):
        return "score" in sample and sample.get("pred") != "Failed"

    def process_sample(self, sample, cfg):
        if sample.get("fail_once") and not sample.get("attempted"):
            sample["attempted"] = True
            return None
        result = dict(sample)
        result["pred"] = "ok"
        result["score"] = 1.0
        return result


class RunnerResultsTests(unittest.TestCase):
    def test_stable_result_paths_partition_by_baseline_dir(self):
        cfg = OmegaConf.create({
            "benchmarks": {"name": "mmlongbench", "qa_model_name": "Qwen3-VL-8B-Instruct"},
            "baselines": {"name": "bm25", "top_k": 3, "chunk_size": 150, "chunk_overlap": 0},
        })

        self.assertEqual(
            str(build_results_file(cfg)),
            str(RESULTS_ROOT / "mmlongbench/bm25/res_top_k_3_chunk_size_150_chunk_overlap_0_Qwen3_VL_8B_Instruct.jsonl"),
        )

    def test_parameter_changes_produce_different_filenames(self):
        cfg_a = OmegaConf.create({
            "benchmarks": {"name": "mmlongbench", "qa_model_name": "model"},
            "baselines": {"name": "bm25", "top_k": 3, "chunk_size": 150, "chunk_overlap": 0},
        })
        cfg_b = OmegaConf.create({
            "benchmarks": {"name": "mmlongbench", "qa_model_name": "model"},
            "baselines": {"name": "bm25", "top_k": 5, "chunk_size": 150, "chunk_overlap": 0},
        })

        self.assertNotEqual(build_results_file(cfg_a), build_results_file(cfg_b))

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
            run_pending(adapter, cfg, samples, output_path)
            run_pending(adapter, cfg, samples, output_path)
            written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual([sample["id"] for sample in written], ["b", "a"])

    def test_generate_reports_marks_missing_metrics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dir = root / "mmlongbench" / "image"
            run_dir.mkdir(parents=True)
            results_file = run_dir / "res_model.jsonl"
            metrics_file, manifest_file = sidecar_paths(results_file)
            write_json(metrics_file, {"overall_acc": 0.8, "sample_count": 2, "completed_count": 2, "failed_count": 0})
            write_json(manifest_file, {
                "benchmark": "mmlongbench",
                "baseline": "image",
                "qa_model_name": "model",
                "parameters": {},
                "results_file": str(results_file),
                "metrics_file": str(metrics_file),
                "generated_at": "now",
            })
            missing_dir = root / "longdocurl" / "ocr"
            missing_dir.mkdir(parents=True)
            missing_results = missing_dir / "res_model.jsonl"
            _, missing_manifest = sidecar_paths(missing_results)
            write_json(missing_manifest, {
                "benchmark": "longdocurl",
                "baseline": "ocr",
                "qa_model_name": "model",
                "parameters": {},
                "results_file": str(missing_results),
                "metrics_file": str(missing_dir / "res_model.metrics.json"),
                "generated_at": "now",
            })

            report = generate_reports(root)

        statuses = {run["benchmark"]: run["status"] for run in report["runs"]}
        self.assertEqual(statuses["mmlongbench"], "ok")
        self.assertEqual(statuses["longdocurl"], "missing_metrics")


if __name__ == "__main__":
    unittest.main()
