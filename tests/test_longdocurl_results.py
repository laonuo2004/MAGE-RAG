import unittest
from pathlib import Path

from omegaconf import OmegaConf

from benchmarks.adapters import LongDocURLAdapter
from benchmarks.utils.results_utils import build_results_file


class LongDocURLResultsTests(unittest.TestCase):
    def test_default_results_file_uses_root_results_jsonl(self):
        cfg = OmegaConf.create({
            "baselines": {"name": "image"},
            "benchmarks": {"name": "longdocurl", "qa_model_name": "Qwen/Qwen2.5-VL-7B-Instruct"},
        })

        self.assertEqual(
            str(build_results_file(cfg)),
            str(Path.cwd() / "results/longdocurl/image/res_Qwen_Qwen2_5_VL_7B_Instruct.jsonl"),
        )

    def test_default_results_file_includes_m3docrag_top_k(self):
        cfg = OmegaConf.create({
            "baselines": {"name": "m3docrag", "params": {"top_k": 5}},
            "benchmarks": {"name": "longdocurl", "qa_model_name": "Qwen/Qwen2.5-VL-7B-Instruct"},
        })

        self.assertEqual(
            str(build_results_file(cfg)),
            str(Path.cwd() / "results/longdocurl/m3docrag/res_top_k_5_Qwen_Qwen2_5_VL_7B_Instruct.jsonl"),
        )

    def test_magerag_results_file_includes_only_sweep_params(self):
        cfg = OmegaConf.create({
            "baselines": {
                "name": "magerag",
                "params": {"top_k": 1},
                "controller": {"watchdog_iterations": 6},
            },
            "benchmarks": {"name": "longdocurl", "qa_model_name": "Qwen/Qwen3-VL-8B-Instruct"},
        })

        result_file = build_results_file(cfg)

        self.assertEqual(
            str(result_file),
            str(Path.cwd() / "results/longdocurl/magerag/res_top_k_1_Qwen_Qwen3_VL_8B_Instruct.jsonl"),
        )
        self.assertLess(len(result_file.name), 120)

    def test_parse_concise_answer_keeps_bare_type_name_as_string(self):
        self.assertEqual(LongDocURLAdapter.parse_concise_answer("int"), "int")

    def test_parse_concise_answer_parses_json_serializable_literals(self):
        self.assertEqual(LongDocURLAdapter.parse_concise_answer("215"), 215)
        self.assertEqual(LongDocURLAdapter.parse_concise_answer("['A', 'B']"), ["A", "B"])
        self.assertCountEqual(LongDocURLAdapter.parse_concise_answer("{'A', 'B'}"), ["A", "B"])

    def test_longdocurl_score_formats(self):
        self.assertEqual(LongDocURLAdapter.score("12", "12", "Integer"), 1.0)
        self.assertEqual(LongDocURLAdapter.score("12.5", "12.5", "Float"), 1.0)
        self.assertEqual(LongDocURLAdapter.score("Alpha", "alpha", "String"), 1.0)
        self.assertEqual(LongDocURLAdapter.score(["A", "B"], ["A", "B"], "List"), 1.0)


if __name__ == "__main__":
    unittest.main()
