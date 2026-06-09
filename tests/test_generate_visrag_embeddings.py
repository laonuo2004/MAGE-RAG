import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmarks.scripts import generate_visrag_embeddings as visrag_script


class GenerateVisRAGEmbeddingsTests(unittest.TestCase):
    def test_retry_failed_filters_doc_and_question_ids(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            failed_path = Path(tmp_dir) / 'failed.jsonl'
            failed_path.write_text(
                json.dumps({'status': 'failed', 'kind': 'pdf', 'key': 'doc-a'}) + '\n'
                + json.dumps({'status': 'failed', 'kind': 'question', 'key': 'q2'}) + '\n'
                + json.dumps({'status': 'generated', 'kind': 'pdf', 'key': 'doc-b'}) + '\n',
                encoding='utf-8',
            )
            args = SimpleNamespace(
                benchmark='longdocurl',
                retry_failed=str(failed_path),
                doc_id=None,
                question_id=None,
                limit=None,
            )
            original_load_samples = visrag_script.load_samples
            try:
                visrag_script.load_samples = lambda _: [
                    {'doc_no': 'doc-a', 'question_id': 'q1', 'question': 'one'},
                    {'doc_no': 'doc-b', 'question_id': 'q2', 'question': 'two'},
                    {'doc_no': 'doc-c', 'question_id': 'q3', 'question': 'three'},
                ]

                selected = visrag_script.selected_samples(args)
            finally:
                visrag_script.load_samples = original_load_samples

        self.assertEqual([sample['question_id'] for sample in selected], ['q1', 'q2'])

    def test_encode_questions_keep_going_records_failed_batch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = SimpleNamespace(
                benchmark='mmlongbench',
                batch_size=2,
                overwrite=True,
                keep_going=True,
                record_traceback=False,
                manifest_dir=tmp_dir,
                failed_output=None,
                mode='question',
            )
            original_encode_query_batch = visrag_script.encode_query_batch
            original_question_path = visrag_script.visrag_question_embeddings_path
            try:
                visrag_script.encode_query_batch = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('boom'))
                visrag_script.visrag_question_embeddings_path = lambda benchmark, question_id: Path(tmp_dir) / f'{question_id}.safetensors'
                failures = visrag_script.encode_questions(
                    args,
                    object(),
                    object(),
                    [
                        {'doc_id': 'doc-a', 'question_id': 'q1', 'question': 'one'},
                        {'doc_id': 'doc-a', 'question_id': 'q2', 'question': 'two'},
                    ],
                )
            finally:
                visrag_script.encode_query_batch = original_encode_query_batch
                visrag_script.visrag_question_embeddings_path = original_question_path

            failed_records = [json.loads(line) for line in (Path(tmp_dir) / 'question_failed.jsonl').read_text().splitlines()]
            manifest_records = [json.loads(line) for line in (Path(tmp_dir) / 'question_manifest.jsonl').read_text().splitlines()]

        self.assertEqual(failures, 2)
        self.assertEqual([record['key'] for record in failed_records], ['q1', 'q2'])
        self.assertEqual([record['status'] for record in manifest_records], ['failed', 'failed'])
        self.assertEqual(failed_records[0]['error'], 'boom')

    def test_encode_questions_writes_generated_manifest(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = SimpleNamespace(
                benchmark='mmlongbench',
                batch_size=2,
                overwrite=True,
                keep_going=False,
                record_traceback=False,
                manifest_dir=tmp_dir,
                failed_output=None,
                mode='question',
            )
            original_encode_query_batch = visrag_script.encode_query_batch
            original_question_path = visrag_script.visrag_question_embeddings_path
            try:
                visrag_script.encode_query_batch = lambda *args, **kwargs: torch.tensor([[1.0, 0.0]])
                visrag_script.visrag_question_embeddings_path = lambda benchmark, question_id: Path(tmp_dir) / f'{question_id}.safetensors'
                failures = visrag_script.encode_questions(
                    args,
                    object(),
                    object(),
                    [{'doc_id': 'doc-a', 'question_id': 'q1', 'question': 'one'}],
                )
            finally:
                visrag_script.encode_query_batch = original_encode_query_batch
                visrag_script.visrag_question_embeddings_path = original_question_path

            manifest_records = [json.loads(line) for line in (Path(tmp_dir) / 'question_manifest.jsonl').read_text().splitlines()]

        self.assertEqual(failures, 0)
        self.assertEqual(manifest_records[0]['status'], 'generated')
        self.assertEqual(manifest_records[0]['shape'], [2])

    def test_rebuild_minicpm_rope_cache_repairs_non_finite_buffers(self):
        class Rotary:
            dim = 4
            base = 10000.0
            max_position_embeddings = 8

            def __init__(self):
                self.cos_cached = torch.full((8, 4), float('nan'))
                self.sin_cached = torch.full((8, 4), float('nan'))

            def register_buffer(self, name, value, persistent=False):
                setattr(self, name, value)

            def _set_cos_sin_cache(self, seq_len, device, dtype):
                positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
                freqs = torch.outer(positions, self.inv_freq)
                emb = torch.cat((freqs, freqs), dim=-1)
                self.cos_cached = emb.cos().to(dtype)
                self.sin_cached = emb.sin().to(dtype)

        rotary = Rotary()
        model = SimpleNamespace(
            llm=SimpleNamespace(
                model=SimpleNamespace(
                    layers=[SimpleNamespace(self_attn=SimpleNamespace(rotary_emb=rotary))]
                )
            )
        )

        visrag_script.rebuild_minicpm_rope_cache(model)

        self.assertTrue(torch.isfinite(rotary.inv_freq).all())
        self.assertTrue(torch.isfinite(rotary.cos_cached).all())
        self.assertTrue(torch.isfinite(rotary.sin_cached).all())
        self.assertAlmostEqual(rotary.inv_freq[0].item(), 1.0)


if __name__ == '__main__':
    unittest.main()
