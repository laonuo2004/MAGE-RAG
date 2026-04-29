import json
import os
import tempfile
import unittest

from omegaconf import OmegaConf

from benchmarks.mmlongbench.run_api import (
    append_result,
    build_default_results_file,
    read_jsonl_results,
)


class MMLongBenchResultsTests(unittest.TestCase):
    def test_default_results_file_uses_jsonl(self):
        cfg = OmegaConf.create({
            "baselines": {"name": "image"},
            "benchmarks": {
                "results_dir": "/tmp/mmlongbench-results",
                "qa_model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
            },
        })

        self.assertEqual(
            build_default_results_file(cfg, cfg.benchmarks),
            "/tmp/mmlongbench-results/res_image_Qwen_Qwen2.5_VL_7B_Instruct.jsonl",
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


if __name__ == "__main__":
    unittest.main()
