import unittest

from omegaconf import OmegaConf

from benchmarks.results import RESULTS_ROOT
from benchmarks.longdocurl.eval.api_models.eval_api_models import build_default_results_file, parse_concise_answer
from utils.config_utils import require_config_value


class LongDocURLResultsTests(unittest.TestCase):
    def test_default_results_file_keeps_existing_baseline_name(self):
        cfg = OmegaConf.create({
            "baselines": {"name": "image"},
            "benchmarks": {"name": "longdocurl", "qa_model_name": "Qwen/Qwen2.5-VL-7B-Instruct"},
        })
        benchmark_cfg = require_config_value(cfg, "benchmarks")

        self.assertEqual(
            build_default_results_file(cfg, benchmark_cfg),
            str(RESULTS_ROOT / "longdocurl/image/res_Qwen_Qwen2_5_VL_7B_Instruct.jsonl"),
        )

    def test_default_results_file_includes_m3docrag_top_k(self):
        cfg = OmegaConf.create({
            "baselines": {"name": "m3docrag", "top_k": 5},
            "benchmarks": {"name": "longdocurl", "qa_model_name": "Qwen/Qwen2.5-VL-7B-Instruct"},
        })
        benchmark_cfg = require_config_value(cfg, "benchmarks")

        self.assertEqual(
            build_default_results_file(cfg, benchmark_cfg),
            str(RESULTS_ROOT / "longdocurl/m3docrag/res_top_k_5_Qwen_Qwen2_5_VL_7B_Instruct.jsonl"),
        )

    def test_parse_concise_answer_keeps_bare_type_name_as_string(self):
        self.assertEqual(parse_concise_answer("int"), "int")

    def test_parse_concise_answer_parses_json_serializable_literals(self):
        self.assertEqual(parse_concise_answer("215"), 215)
        self.assertEqual(parse_concise_answer("['A', 'B']"), ["A", "B"])
        self.assertCountEqual(parse_concise_answer("{'A', 'B'}"), ["A", "B"])


if __name__ == "__main__":
    unittest.main()
