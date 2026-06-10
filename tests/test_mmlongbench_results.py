import json
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from benchmarks.adapters import MMLongBenchAdapter
from benchmarks.runner import compact_results_file, merge_existing_samples, run_pending, successful_results
from benchmarks.utils.results_utils import append_jsonl, build_results_file, read_jsonl


class MMLongBenchResultsTests(unittest.TestCase):
    def test_default_results_file_uses_root_results_jsonl(self):
        cfg = OmegaConf.create({
            "baselines": {"name": "image"},
            "benchmarks": {
                "name": "mmlongbench",
                "qa_model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
            },
        })

        self.assertEqual(
            str(build_results_file(cfg)),
            str(Path.cwd() / "results/mmlongbench/image/res_Qwen_Qwen2_5_VL_7B_Instruct.jsonl"),
        )

    def test_append_read_and_compact_jsonl_results(self):
        adapter = MMLongBenchAdapter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.jsonl"
            append_jsonl({"doc_id": "d1", "question": "q1", "answer": "a", "answer_format": "Str", "pred": "Failed to extract", "score": 0.0}, output_path)
            append_jsonl({"doc_id": "d2", "question": "q2", "answer": "a", "answer_format": "Str", "pred": "ok", "score": 0.0}, output_path)
            output_path.write_text(output_path.read_text(encoding="utf-8") + "{bad json}\n", encoding="utf-8")

            raw = read_jsonl(output_path)
            compact_results_file(adapter, output_path)
            compacted = read_jsonl(output_path)

        self.assertEqual(len(raw), 2)
        self.assertEqual(compacted, [{"doc_id": "d2", "question": "q2", "answer": "a", "answer_format": "Str", "pred": "ok", "score": 0.0}])

    def test_merge_existing_uses_adapter_key(self):
        adapter = MMLongBenchAdapter()
        sample = {"doc_id": "d1", "question": "q", "answer": "a", "answer_format": "Str"}
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.jsonl"
            append_jsonl({**sample, "pred": "a", "score": 1.0}, output_path)
            merged = merge_existing_samples(adapter, [sample], output_path)
            successful = successful_results(adapter, output_path)

        self.assertEqual(merged[0]["score"], 1.0)
        self.assertEqual(successful, [{**sample, "pred": "a", "score": 1.0}])

    def test_run_pending_does_not_append_failed_sample(self):
        class Adapter(MMLongBenchAdapter):
            async def process_sample_async(self, sample, cfg, context_builder, client, *, local_executor=None):
                return None

        samples = [{"doc_id": "d1", "question": "q?", "answer": "a", "answer_format": "Str"}]
        cfg = OmegaConf.create({"benchmarks": {"local_workers": 1}})
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.jsonl"
            run_pending(Adapter(), cfg, samples, output_path, object(), object())

        self.assertFalse(output_path.exists())
        self.assertNotIn("score", samples[0])

    def test_score_and_f1_handle_not_answerable(self):
        adapter = MMLongBenchAdapter()
        samples = [
            {"answer": "a", "pred": "a", "score": adapter_score("a", "a", "Str")},
            {"answer": "Not answerable", "pred": "Not answerable", "score": 1.0},
        ]

        metrics = adapter.build_metrics(samples, Path("result.jsonl"))

        self.assertEqual(metrics["overall_acc"], 1.0)
        self.assertEqual(metrics["overall_f1"], 1.0)


def adapter_score(answer, pred, fmt):
    from benchmarks.adapters import MMLongBenchAdapter

    return MMLongBenchAdapter.score(answer, pred, fmt)


if __name__ == "__main__":
    unittest.main()
