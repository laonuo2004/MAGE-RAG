import unittest

from omegaconf import OmegaConf

from benchmarks.longdocurl.eval.api_models.eval_api_models import (
    BENCHMARK_ROOT,
    build_default_results_file,
)
from utils.config_utils import require_config_value


class LongDocURLResultsTests(unittest.TestCase):
    def test_default_results_file_keeps_existing_baseline_name(self):
        cfg = OmegaConf.create({
            "baselines": {"name": "image"},
            "benchmarks": {"qa_model_name": "Qwen/Qwen2.5-VL-7B-Instruct"},
        })
        benchmark_cfg = require_config_value(cfg, "benchmarks")

        self.assertEqual(
            build_default_results_file(cfg, benchmark_cfg),
            str(BENCHMARK_ROOT / "evaluation_results/api_models/results_image_Qwen_Qwen2.5_VL_7B_Instruct.jsonl"),
        )

    def test_default_results_file_includes_m3docrag_top_k(self):
        cfg = OmegaConf.create({
            "baselines": {"name": "m3docrag", "top_k": 5},
            "benchmarks": {"qa_model_name": "Qwen/Qwen2.5-VL-7B-Instruct"},
        })
        benchmark_cfg = require_config_value(cfg, "benchmarks")

        self.assertEqual(
            build_default_results_file(cfg, benchmark_cfg),
            str(BENCHMARK_ROOT / "evaluation_results/api_models/results_m3docrag_top_k_5_Qwen_Qwen2.5_VL_7B_Instruct.jsonl"),
        )


if __name__ == "__main__":
    unittest.main()
