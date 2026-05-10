import json
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from benchmarks.adapters import LongDocURLAdapter, MMLongBenchAdapter


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


if __name__ == "__main__":
    unittest.main()
