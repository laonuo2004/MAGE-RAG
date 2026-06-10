import unittest
import base64
import json
import os
import tempfile
from io import BytesIO

from omegaconf import OmegaConf
from safetensors.torch import save_file
import torch

import baselines.m3docrag_iterate as m3docrag_iterate
import baselines.m3docrag_iterate_query as m3docrag_iterate_query
from baselines.bm25 import BM25ContextBuilder
from baselines.colbertv2 import ColBERTv2ContextBuilder
from baselines.evisrag import EVisRAGContextBuilder
from benchmarks.utils.document_preprocess import (
    colbertv2_doc_cache_variant,
    colbertv2_query_cache_variant,
    encode_pil_image_to_base64,
)
from baselines.wrapper import build_context_builder
from baselines.base import build_context_summary, build_logical_cost, build_retrieval_metadata
from benchmarks import wrapper
from benchmarks.utils.data_utils import (
    colbertv2_cache_root,
    colpali_pdf_embeddings_path,
    colpali_question_embeddings_path,
    visrag_pdf_embeddings_path,
    visrag_question_embeddings_path,
)


class ContextBuilderTests(unittest.TestCase):
    def test_metadata_helpers_build_stable_empty_structures(self):
        logical_cost = build_logical_cost()
        retrieval = build_retrieval_metadata()
        context_summary = build_context_summary()

        self.assertEqual(logical_cost["num_llm_calls"], 0)
        self.assertEqual(logical_cost["num_retriever_calls"], 0)
        self.assertEqual(logical_cost["num_input_images"], 0)
        self.assertEqual(retrieval["retrieved_items"], [])
        self.assertEqual(retrieval["initial_retrieved_pages"], [])
        self.assertEqual(context_summary["num_context_pages"], 0)
        self.assertEqual(context_summary["page_ids"], [])

    def test_build_context_builder_selects_image_and_ocr(self):
        image_builder = build_context_builder(OmegaConf.create({'baselines': {'name': 'image'}}))
        ocr_builder = build_context_builder(OmegaConf.create({'baselines': {'name': 'ocr'}}))
        bm25_builder = build_context_builder(OmegaConf.create({'baselines': {'name': 'bm25'}}))
        colbert_builder = build_context_builder(OmegaConf.create({
            'baselines': {
                'name': 'colbertv2',
                'params': {'top_k': 1, 'chunk_size': 8, 'chunk_overlap': 0},
                'checkpoint': 'dummy-checkpoint',
            }
        }))

        self.assertEqual(image_builder.name, 'image')
        self.assertEqual(ocr_builder.name, 'ocr')
        self.assertEqual(bm25_builder.name, 'bm25')
        iterate_builder = build_context_builder(OmegaConf.create({
            'baselines': {
                'name': 'm3docrag-iterate',
                'params': {'max_iterations': 5, 'evaluator_model_name': 'eval-model'},
            }
        }))

        self.assertEqual(colbert_builder.name, 'colbertv2')
        iterate_query_builder = build_context_builder(OmegaConf.create({
            'baselines': {
                'name': 'm3docrag-iterate-query',
                'params': {'max_iterations': 5, 'evaluator_model_name': 'eval-model'},
            }
        }))

        self.assertEqual(iterate_builder.name, 'm3docrag-iterate')
        self.assertEqual(iterate_query_builder.name, 'm3docrag-iterate-query')
        evisrag_builder = build_context_builder(OmegaConf.create({
            'baselines': {'name': 'evisrag', 'params': {'top_k': 3}},
        }))
        self.assertEqual(evisrag_builder.name, 'evisrag')

    def test_longdocurl_image_context_matches_legacy_prompt_shape(self):
        png_bytes = base64.b64decode(
            'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII='
        )
        builder = build_context_builder(OmegaConf.create({
            'baselines': {'name': 'image'},
            'benchmarks': {'name': 'longdocurl'},
        }))

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_paths = [os.path.join(tmp_dir, 'page1.png'), os.path.join(tmp_dir, 'page2.png')]
            for image_path in image_paths:
                with open(image_path, 'wb') as f:
                    f.write(png_bytes)
            sample = {'question': 'What is shown?', 'images': image_paths}
            messages = builder.build('longdocurl', sample)

        self.assertIsInstance(messages, list)
        self.assertEqual(messages[0]['role'], 'user')
        content = messages[0]['content']
        self.assertEqual(
            content[0],
            {
                'type': 'text',
                'text': (
                    'You are an expert in visual document question-answering, please answer our questions based on the given images.\n'
                    'Following is our question: \n'
                    '<question>What is shown?</question>\n'
                ),
            },
        )
        self.assertEqual(content[1], {'type': 'text', 'text': 'Below is the 1-th image (total 2 images).\n'})
        self.assertEqual(content[2]['type'], 'image_url')
        self.assertTrue(content[2]['image_url']['url'].startswith('data:image/png;base64,'))
        self.assertEqual(messages.metadata['context_builder'], 'image')
        self.assertEqual(messages.metadata['retrieval']['final_context_pages'], [0, 1])
        self.assertEqual(messages.metadata['context_summary']['num_context_pages'], 2)
        self.assertEqual(messages.metadata['logical_cost']['num_input_images'], 2)

    def test_wrapper_routes_longdocurl_ocr_baseline(self):
        captured = []
        original_runner = wrapper.run_benchmark_with_adapter
        try:
            wrapper.run_benchmark_with_adapter = lambda cfg, adapter: captured.append((cfg, adapter.name))
            wrapper.run_benchmark(OmegaConf.create({
                'benchmarks': {'name': 'longdocurl', 'qa_file': 'qa.jsonl', 'model_name': 'model'},
                'baselines': {'name': 'ocr'},
            }))
        finally:
            wrapper.run_benchmark_with_adapter = original_runner

        self.assertEqual(captured[0][0].baselines.name, 'ocr')
        self.assertEqual(captured[0][1], 'longdocurl')

    def test_mmlongbench_ocr_reads_preprocessed_json_pages(self):
        builder = build_context_builder(OmegaConf.create({
            'baselines': {'name': 'ocr'},
            'benchmarks': {'name': 'mmlongbench'},
        }))

        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = OmegaConf.create({
                'benchmarks': {
                    'tmp_dir': tmp_dir,
                    'ocr_json_dir': os.path.join(tmp_dir, 'pdf_jsons'),
                    'max_pages': 3,
                }
            })
            builder.cfg = cfg
            page_dir = os.path.join(tmp_dir, 'pdf_jsons', 'sample')
            os.makedirs(page_dir)
            with open(os.path.join(page_dir, 'page_0001.json'), 'w', encoding='utf-8') as f:
                json.dump({'text': 'first page text'}, f)
            with open(os.path.join(page_dir, 'page_0002.json'), 'w', encoding='utf-8') as f:
                json.dump({'text': 'second page text'}, f)

            messages = builder.build('mmlongbench', {'doc_id': 'sample.pdf', 'question': 'Question?'})

        prompt = messages[0]['content'][0]['text']
        self.assertEqual(messages[0]['content'][0]['type'], 'text')
        self.assertIn('[Page 1]\nfirst page text', prompt)
        self.assertIn('[Page 2]\nsecond page text', prompt)
        self.assertNotIn('[Page 3]', prompt)

    def test_mmlongbench_image_reads_preprocessed_png_pages(self):
        from PIL import Image

        buffer = BytesIO()
        Image.new('RGB', (1, 1), color='white').save(buffer, format='PNG')
        png_bytes = buffer.getvalue()
        builder = build_context_builder(OmegaConf.create({
            'baselines': {'name': 'image'},
            'benchmarks': {'name': 'mmlongbench'},
        }))

        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = OmegaConf.create({
                'benchmarks': {
                    'tmp_dir': tmp_dir,
                    'pdf_png_dir': os.path.join(tmp_dir, 'pdf_pngs'),
                    'max_pages': 3,
                    'resolution': 144,
                }
            })
            builder.cfg = cfg
            page_dir = os.path.join(tmp_dir, 'pdf_pngs', 'sample')
            os.makedirs(page_dir)
            for page_no in (1, 2):
                with open(os.path.join(page_dir, f'page_{page_no:04d}_dpi144.png'), 'wb') as f:
                    f.write(png_bytes)

            messages = builder.build('mmlongbench', {'doc_id': 'sample.pdf', 'question': 'What is shown?'})

        content = messages[0]['content']
        self.assertEqual(content[0], {'type': 'text', 'text': 'What is shown?'})
        self.assertEqual(len(content), 3)
        self.assertTrue(content[1]['image_url']['url'].startswith('data:image/jpeg;base64,'))
        self.assertTrue(content[2]['image_url']['url'].startswith('data:image/jpeg;base64,'))

    def test_image_base64_utility_encodes_pil_image(self):
        from PIL import Image

        encoded = encode_pil_image_to_base64(Image.new('RGBA', (1, 1), color='white'))

        self.assertIsInstance(encoded, str)
        self.assertGreater(len(base64.b64decode(encoded)), 0)

    def test_m3docrag_mmlongbench_retrieves_expected_top_page(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_embeddings(
                tmp_dir,
                benchmark_name='mmlongbench',
                doc_stem='sample',
                question_id='q1',
                doc_embeddings=torch.tensor([
                    [[1.0, 0.0]],
                    [[0.0, 1.0]],
                    [[3.0, 0.0]],
                ]),
                query_embedding=torch.tensor([[1.0, 0.0]]),
            )
            page_dir = os.path.join(tmp_dir, 'pdf_pngs', 'sample')
            os.makedirs(page_dir)
            for page_no in (1, 2, 3):
                Image.new('RGB', (1, 1), color='white').save(
                    os.path.join(page_dir, f'page_{page_no:04d}_dpi144.png')
                )
            builder = build_context_builder(self._m3docrag_cfg(tmp_dir, max_pages=3))

            messages = builder.build(
                'mmlongbench',
                {'doc_id': 'sample.pdf', 'question_id': 'q1', 'question': 'Question?'},
            )

        self.assertEqual(messages.metadata['retrieved_pages'][0]['page_index'], 2)
        self.assertEqual(messages.metadata['retrieved_pages'][0]['page_number'], 3)
        self.assertEqual(messages.metadata['allowed_pages'], [0, 1, 2])
        self.assertEqual(messages.metadata['retrieval']['retrieved_items'][0]['page_index'], 2)
        self.assertEqual(messages.metadata['retrieval']['initial_retrieved_pages'], [2])
        self.assertEqual(messages.metadata['context_summary']['num_context_pages'], 1)
        self.assertEqual(messages.metadata['logical_cost']['num_retriever_calls'], 1)
        self.assertEqual(len(messages[0]['content']), 2)

    def test_bm25_metadata_exposes_unified_retrieval_context_and_cost(self):
        builder = BM25ContextBuilder(OmegaConf.create({
            'baselines': {
                'name': 'bm25',
                'params': {'top_k': 2, 'chunk_size': 8, 'chunk_overlap': 0},
                'tokenizer': 'regex',
            },
        }))

        metadata = builder._metadata(
            [
                {'rank': 1, 'chunk_id': 3, 'chunk_index': 0, 'page_index': 2, 'page_number': 3, 'score': 1.5, 'text': 'alpha'},
                {'rank': 2, 'chunk_id': 4, 'chunk_index': 1, 'page_index': 2, 'page_number': 3, 'score': 0.5, 'text': 'beta'},
            ],
            [0, 1, 2],
        )

        self.assertEqual(metadata['retrieval']['retrieved_items'][0]['chunk_id'], 3)
        self.assertEqual(metadata['retrieval']['initial_retrieved_pages'], [2])
        self.assertEqual(metadata['context_summary']['num_context_pages'], 1)
        self.assertEqual(metadata['context_summary']['num_text_units'], 2)
        self.assertEqual(metadata['logical_cost']['num_retriever_calls'], 1)
        self.assertEqual(metadata['logical_cost']['num_retrieved_chunks'], 2)

    def test_colbertv2_metadata_exposes_unified_retrieval_context_and_cost(self):
        builder = ColBERTv2ContextBuilder(OmegaConf.create({
            'baselines': {
                'name': 'colbertv2',
                'checkpoint': 'dummy-checkpoint',
                'params': {'top_k': 2, 'chunk_size': 8, 'chunk_overlap': 0},
            },
        }))

        metadata = builder._metadata(
            [
                {
                    'rank': 1,
                    'chunk_id': 10,
                    'chunk_index': 0,
                    'score': 2.0,
                    'text': 'alpha beta',
                    'covered_page_indices': [1, 2],
                    'covered_page_numbers': [2, 3],
                },
            ],
            [1, 2],
            'doc',
            'query',
        )

        self.assertEqual(metadata['retrieval']['retrieved_items'][0]['chunk_id'], 10)
        self.assertEqual(metadata['retrieval']['initial_retrieved_pages'], [1, 2])
        self.assertEqual(metadata['context_summary']['page_ids'], [1, 2])
        self.assertEqual(metadata['logical_cost']['num_retrieved_chunks'], 1)

    def test_evisrag_metadata_exposes_unified_retrieval_context_and_cost(self):
        builder = EVisRAGContextBuilder(OmegaConf.create({
            'baselines': {'name': 'evisrag', 'params': {'top_k': 3, 'max_images': 2}},
        }))

        metadata = builder._metadata(
            [
                {'page_index': 4, 'page_number': 5, 'score': 3.0},
                {'page_index': 1, 'page_number': 2, 'score': 2.0},
                {'page_index': 2, 'page_number': 3, 'score': 1.0},
            ],
            [1, 2, 4],
            'pdf.safetensors',
            'question.safetensors',
        )

        self.assertEqual(metadata['retrieval']['initial_retrieved_pages'], [4, 1, 2])
        self.assertEqual(metadata['retrieval']['final_context_pages'], [4, 1])
        self.assertEqual(metadata['context_summary']['num_image_units'], 2)
        self.assertEqual(metadata['logical_cost']['num_retriever_calls'], 1)

    def test_m3docrag_longdocurl_images_mask_blocks_out_of_range_page(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_embeddings(
                tmp_dir,
                benchmark_name='longdocurl',
                doc_stem='123456',
                question_id='q1',
                doc_embeddings=torch.tensor([
                    [[1.0, 0.0]],
                    [[9.0, 0.0]],
                ]),
                query_embedding=torch.tensor([[1.0, 0.0]]),
            )
            page_dir = os.path.join(tmp_dir, 'longdoc_images', '1234')
            os.makedirs(page_dir)
            for page_index in (0, 1):
                Image.new('RGB', (1, 1), color='white').save(
                    os.path.join(page_dir, f'123456_{page_index}.png')
                )
            builder = build_context_builder(self._m3docrag_cfg(tmp_dir))

            messages = builder.build(
                'longdocurl',
                {
                    'doc_no': '123456',
                    'question_id': 'q1',
                    'question': 'Question?',
                    'start_end_idx': [1, 2],
                    'images': [os.path.join(page_dir, '123456_0.png')],
                },
            )

        self.assertEqual(messages.metadata['allowed_pages'], [0])
        self.assertEqual(messages.metadata['retrieved_pages'][0]['page_index'], 0)

    def test_m3docrag_longdocurl_ignores_start_end_idx_for_page_mask(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_embeddings(
                tmp_dir,
                benchmark_name='longdocurl',
                doc_stem='123456',
                question_id='q1',
                doc_embeddings=torch.tensor([
                    [[1.0, 0.0]],
                    [[9.0, 0.0]],
                ]),
                query_embedding=torch.tensor([[1.0, 0.0]]),
            )
            page_dir = os.path.join(tmp_dir, 'longdoc_images', '1234')
            os.makedirs(page_dir)
            for page_index in (0, 1):
                Image.new('RGB', (1, 1), color='white').save(
                    os.path.join(page_dir, f'123456_{page_index}.png')
                )
            builder = build_context_builder(self._m3docrag_cfg(tmp_dir))

            messages = builder.build(
                'longdocurl',
                {
                    'doc_no': '123456',
                    'question_id': 'q1',
                    'question': 'Question?',
                    'start_end_idx': [2, 2],
                    'images': [os.path.join(page_dir, '123456_0.png')],
                },
            )

        self.assertEqual(messages.metadata['allowed_pages'], [0])
        self.assertEqual(messages.metadata['retrieved_pages'][0]['page_index'], 0)

    def test_m3docrag_mmlongbench_max_pages_mask_blocks_out_of_range_page(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_embeddings(
                tmp_dir,
                benchmark_name='mmlongbench',
                doc_stem='sample',
                question_id='q1',
                doc_embeddings=torch.tensor([
                    [[1.0, 0.0]],
                    [[9.0, 0.0]],
                ]),
                query_embedding=torch.tensor([[1.0, 0.0]]),
            )
            page_dir = os.path.join(tmp_dir, 'pdf_pngs', 'sample')
            os.makedirs(page_dir)
            for page_no in (1, 2):
                Image.new('RGB', (1, 1), color='white').save(
                    os.path.join(page_dir, f'page_{page_no:04d}_dpi144.png')
                )
            builder = build_context_builder(self._m3docrag_cfg(tmp_dir, max_pages=1))

            messages = builder.build(
                'mmlongbench',
                {'doc_id': 'sample.pdf', 'question_id': 'q1', 'question': 'Question?'},
            )

        self.assertEqual(messages.metadata['allowed_pages'], [0])
        self.assertEqual(messages.metadata['retrieved_pages'][0]['page_index'], 0)

    def test_m3docrag_requires_explicit_top_k(self):
        with self.assertRaisesRegex(ValueError, 'top_k'):
            build_context_builder(OmegaConf.create({'baselines': {'name': 'm3docrag'}}))

    def test_evisrag_mmlongbench_retrieves_top3_with_evidence_prompt(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_visrag_embeddings(
                tmp_dir,
                benchmark_name='mmlongbench',
                doc_stem='sample',
                question_id='q1',
                doc_embeddings=torch.tensor([
                    [1.0, 0.0],
                    [0.0, 4.0],
                    [3.0, 0.0],
                    [2.0, 0.0],
                ]),
                query_embedding=torch.tensor([1.0, 0.0]),
            )
            page_dir = os.path.join(tmp_dir, 'pdf_pngs', 'sample')
            os.makedirs(page_dir)
            for page_no in (1, 2, 3, 4):
                Image.new('RGB', (1, 1), color='white').save(
                    os.path.join(page_dir, f'page_{page_no:04d}_dpi144.png')
                )
            builder = build_context_builder(self._evisrag_cfg(tmp_dir, max_pages=4))

            messages = builder.build(
                'mmlongbench',
                {'doc_id': 'sample.pdf', 'question_id': 'q1', 'question': 'Question?'},
            )

        self.assertEqual([page['page_index'] for page in messages.metadata['retrieved_pages']], [2, 3, 0])
        self.assertEqual(messages.metadata['allowed_pages'], [0, 1, 2, 3])
        self.assertEqual(messages.metadata['top_k'], 3)
        self.assertEqual(messages.metadata['generation_prompt'], 'evidence_grpo')
        self.assertEqual(len(messages[0]['content']), 4)
        self.assertIn('<evidence></evidence>', messages[0]['content'][0]['text'])
        self.assertIn('Question: Question?', messages[0]['content'][0]['text'])

    def test_evisrag_longdocurl_uses_image_page_mask(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_visrag_embeddings(
                tmp_dir,
                benchmark_name='longdocurl',
                doc_stem='123456',
                question_id='q1',
                doc_embeddings=torch.tensor([
                    [9.0, 0.0],
                    [1.0, 0.0],
                    [8.0, 0.0],
                ]),
                query_embedding=torch.tensor([1.0, 0.0]),
            )
            page_dir = os.path.join(tmp_dir, 'longdoc_images', '1234')
            os.makedirs(page_dir)
            for page_index in (0, 1, 2):
                Image.new('RGB', (1, 1), color='white').save(
                    os.path.join(page_dir, f'123456_{page_index}.png')
                )
            builder = build_context_builder(self._evisrag_cfg(tmp_dir))

            messages = builder.build(
                'longdocurl',
                {
                    'doc_no': '123456',
                    'question_id': 'q1',
                    'question': 'Question?',
                    'images': [os.path.join(page_dir, '123456_1.png')],
                },
            )

        self.assertEqual(messages.metadata['allowed_pages'], [1])
        self.assertEqual(messages.metadata['retrieved_pages'][0]['page_index'], 1)
        self.assertEqual(len(messages[0]['content']), 3)
        self.assertEqual(messages[0]['content'][1]['type'], 'text')
        self.assertEqual(messages[0]['content'][2]['type'], 'image_url')

    def test_m3docrag_iterate_top1_answerable_returns_one_page(self):
        messages = self._build_m3docrag_iterate_mmlongbench(['{"answerable": true, "reason": "enough", "missing_evidence": ""}'])

        self.assertEqual(messages.metadata['candidate_pages'][0]['page_index'], 2)
        self.assertEqual([page['page_index'] for page in messages.metadata['retrieved_pages']], [2])
        self.assertEqual(messages.metadata['stopped_by'], 'answerable')
        self.assertEqual(len(messages.metadata['iteration_trace']), 1)
        self.assertEqual(len(messages[0]['content']), 2)

    def test_m3docrag_iterate_accumulates_until_answerable(self):
        messages = self._build_m3docrag_iterate_mmlongbench([
            '{"answerable": false, "reason": "need more", "missing_evidence": "next page"}',
            '{"answerable": true, "reason": "enough", "missing_evidence": ""}',
        ])

        self.assertEqual([page['page_index'] for page in messages.metadata['retrieved_pages']], [2, 1])
        self.assertEqual(messages.metadata['stopped_by'], 'answerable')
        self.assertEqual(len(messages.metadata['iteration_trace']), 2)
        self.assertEqual(len(messages[0]['content']), 3)

    def test_m3docrag_iterate_invalid_json_falls_back_to_max_iterations(self):
        messages = self._build_m3docrag_iterate_mmlongbench(['not-json'])

        self.assertEqual([page['page_index'] for page in messages.metadata['retrieved_pages']], [2, 1, 0])
        self.assertEqual(messages.metadata['stopped_by'], 'fallback_evaluator_error')
        self.assertIn('error', messages.metadata['iteration_trace'][0])
        self.assertEqual(len(messages[0]['content']), 4)

    def test_m3docrag_iterate_all_false_falls_back_to_max_iterations(self):
        messages = self._build_m3docrag_iterate_mmlongbench([
            '{"answerable": false, "reason": "no", "missing_evidence": "more"}',
            '{"answerable": false, "reason": "no", "missing_evidence": "more"}',
            '{"answerable": false, "reason": "no", "missing_evidence": "more"}',
        ])

        self.assertEqual([page['page_index'] for page in messages.metadata['retrieved_pages']], [2, 1, 0])
        self.assertEqual(messages.metadata['stopped_by'], 'fallback_max_iterations')
        self.assertEqual(len(messages.metadata['iteration_trace']), 3)
        self.assertEqual(len(messages[0]['content']), 4)

    def test_m3docrag_iterate_query_retrieves_next_page_from_search_query(self):
        messages = self._build_m3docrag_iterate_query_mmlongbench(
            [
                '{"answerable": false, "reason": "need linked page", "missing_evidence": "table", "search_query": "table evidence"}',
                '{"answerable": true, "reason": "enough", "missing_evidence": "", "search_query": ""}',
            ],
            [torch.tensor([[0.0, 1.0]])],
        )

        self.assertEqual([page['page_index'] for page in messages.metadata['retrieved_pages']], [2, 1])
        self.assertEqual(messages.metadata['stopped_by'], 'answerable')
        self.assertEqual(messages.metadata['iteration_trace'][0]['search_query'], 'table evidence')
        self.assertEqual(messages.metadata['iteration_trace'][0]['added_page']['page_index'], 1)
        self.assertEqual(messages.metadata['online_colpali']['backbone_path'], '/test/backbone')
        self.assertEqual(len(messages[0]['content']), 3)

    def test_m3docrag_iterate_query_answerable_does_not_load_online_colpali(self):
        messages = self._build_m3docrag_iterate_query_mmlongbench([
            '{"answerable": true, "reason": "enough", "missing_evidence": "", "search_query": ""}',
        ])

        self.assertEqual([page['page_index'] for page in messages.metadata['retrieved_pages']], [2])
        self.assertEqual(messages.metadata['stopped_by'], 'answerable')
        self.assertIsNone(messages.metadata['online_colpali']['resolved_device'])
        self.assertEqual(len(messages.metadata['iteration_trace']), 1)

    def test_m3docrag_iterate_query_invalid_json_returns_current_page(self):
        messages = self._build_m3docrag_iterate_query_mmlongbench(['not-json'])

        self.assertEqual([page['page_index'] for page in messages.metadata['retrieved_pages']], [2])
        self.assertEqual(messages.metadata['stopped_by'], 'fallback_evaluator_error')
        self.assertIn('error', messages.metadata['iteration_trace'][0])

    def test_m3docrag_iterate_query_empty_search_query_returns_current_page(self):
        messages = self._build_m3docrag_iterate_query_mmlongbench([
            '{"answerable": false, "reason": "no", "missing_evidence": "more", "search_query": ""}',
        ])

        self.assertEqual([page['page_index'] for page in messages.metadata['retrieved_pages']], [2])
        self.assertEqual(messages.metadata['stopped_by'], 'fallback_empty_search_query')
        self.assertEqual(messages.metadata['iteration_trace'][0]['search_query'], '')

    def test_m3docrag_iterate_query_loads_online_model_per_search_query(self):
        builder = build_context_builder(OmegaConf.create({
            'baselines': {
                'name': 'm3docrag-iterate-query',
                'params': {'max_iterations': 3, 'evaluator_model_name': 'eval-model'},
                'online_colpali': {
                    'backbone_path': '/test/backbone',
                    'adapter_path': '/test/adapter',
                    'device': 'cpu',
                    'query_batch_size': 1,
                },
            }
        }))
        calls = {'load': 0, 'release': 0}

        class FakeRetrievalModel:
            def encode_queries(self, queries, batch_size, to_cpu):
                return [torch.tensor([[float(len(queries[0])), 0.0]])]

        def fake_load_online_retrieval_model():
            calls['load'] += 1
            return FakeRetrievalModel()

        def fake_release_online_model_memory():
            calls['release'] += 1

        builder._load_online_retrieval_model = fake_load_online_retrieval_model
        builder._release_online_model_memory = fake_release_online_model_memory

        builder._encode_search_query('first')
        builder._encode_search_query('second')

        self.assertEqual(calls, {'load': 2, 'release': 2})

    def test_bm25_mmlongbench_retrieves_matching_chunk(self):
        builder = build_context_builder(OmegaConf.create({
            'baselines': {
                'name': 'bm25',
                'params': {
                    'top_k': 1,
                    'chunk_size': 4,
                    'chunk_overlap': 0,
                },
                'tokenizer': 'regex',
            },
            'benchmarks': {'name': 'mmlongbench'},
        }))

        with tempfile.TemporaryDirectory() as tmp_dir:
            builder.cfg.benchmarks = {
                'tmp_dir': tmp_dir,
                'ocr_json_dir': os.path.join(tmp_dir, 'pdf_jsons'),
                'max_pages': 2,
            }
            page_dir = os.path.join(tmp_dir, 'pdf_jsons', 'sample')
            os.makedirs(page_dir)
            with open(os.path.join(page_dir, 'page_0001.json'), 'w', encoding='utf-8') as f:
                json.dump({'text': 'alpha beta gamma delta'}, f)
            with open(os.path.join(page_dir, 'page_0002.json'), 'w', encoding='utf-8') as f:
                json.dump({'text': 'needle evidence value answer'}, f)

            messages = builder.build(
                'mmlongbench',
                {'doc_id': 'sample.pdf', 'question': 'Where is the needle evidence?'},
            )

        self.assertEqual(messages.metadata['retrieved_chunks'][0]['page_index'], 1)
        self.assertEqual(messages.metadata['retrieved_pages'][0]['page_number'], 2)
        prompt = messages[0]['content'][0]['text']
        self.assertEqual(messages[0]['content'][0]['type'], 'text')
        self.assertIn('[Page 2 | Chunk 1', prompt)
        self.assertIn('needle evidence value answer', prompt)

    def test_bm25_longdocurl_uses_image_page_range(self):
        builder = build_context_builder(OmegaConf.create({
            'baselines': {
                'name': 'bm25',
                'params': {
                    'top_k': 1,
                    'chunk_size': 8,
                    'chunk_overlap': 0,
                },
                'tokenizer': 'regex',
            },
            'benchmarks': {'name': 'longdocurl'},
        }))

        with tempfile.TemporaryDirectory() as tmp_dir:
            ocr_json_dir = os.path.join(tmp_dir, 'pdf_jsons')
            doc_dir = os.path.join(ocr_json_dir, '1234')
            os.makedirs(doc_dir)
            builder.cfg.benchmarks = {
                'ocr_json_dir': ocr_json_dir,
            }
            record = {
                'contents': [
                    {'page_no': 0, 'block_no': 0, 'line_no': 0, 'word_no': 0, 'word': 'needle'},
                    {'page_no': 0, 'block_no': 0, 'line_no': 0, 'word_no': 1, 'word': 'outside'},
                    {'page_no': 1, 'block_no': 0, 'line_no': 0, 'word_no': 0, 'word': 'allowed'},
                    {'page_no': 1, 'block_no': 0, 'line_no': 0, 'word_no': 1, 'word': 'target'},
                ]
            }
            with open(os.path.join(doc_dir, '123456.json'), 'w', encoding='utf-8') as f:
                json.dump(record, f)

            image_path = os.path.join(tmp_dir, '123456_1.png')
            messages = builder.build(
                'longdocurl',
                {
                    'doc_no': '123456',
                    'question': 'needle target',
                    'start_end_idx': [0, 1],
                    'images': [image_path],
                },
            )

        self.assertEqual(messages.metadata['allowed_pages'], [1])
        self.assertEqual(messages.metadata['retrieved_chunks'][0]['page_index'], 1)
        self.assertNotIn('needle outside', messages[0]['content'][0]['text'])
        self.assertIn('allowed target', messages[0]['content'][0]['text'])

    def test_colbertv2_mmlongbench_retrieves_expected_chunk(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            doc_variant = colbertv2_doc_cache_variant('dummy-checkpoint', 8, 0, True, None)
            query_variant = colbertv2_query_cache_variant('dummy-checkpoint')
            doc_variant_dir = colbertv2_cache_root('mmlongbench', 'doc_embeddings') / doc_variant
            query_variant_dir = colbertv2_cache_root('mmlongbench', 'query_embeddings') / query_variant
            meta_variant_dir = colbertv2_cache_root('mmlongbench', 'chunk_metadata') / doc_variant
            os.makedirs(doc_variant_dir, exist_ok=True)
            os.makedirs(query_variant_dir, exist_ok=True)
            os.makedirs(meta_variant_dir, exist_ok=True)

            save_file(
                {
                    'embeddings': torch.tensor([[1.0, 0.0], [5.0, 0.0], [0.0, 1.0]]),
                    'doclens': torch.tensor([1, 1, 1], dtype=torch.int32),
                },
                doc_variant_dir / 'sample.safetensors',
            )
            save_file(
                {'query_embedding': torch.tensor([[1.0, 0.0]])},
                query_variant_dir / 'q1.safetensors',
            )
            with open(meta_variant_dir / 'sample.json', 'w', encoding='utf-8') as f:
                json.dump([
                    {
                        'chunk_id': 0,
                        'chunk_index': 0,
                        'text': 'alpha',
                        'start_page_index': 0,
                        'end_page_index': 0,
                        'start_page_number': 1,
                        'end_page_number': 1,
                        'covered_page_indices': [0],
                        'covered_page_numbers': [1],
                    },
                    {
                        'chunk_id': 1,
                        'chunk_index': 1,
                        'text': 'needle evidence',
                        'start_page_index': 1,
                        'end_page_index': 1,
                        'start_page_number': 2,
                        'end_page_number': 2,
                        'covered_page_indices': [1],
                        'covered_page_numbers': [2],
                    },
                    {
                        'chunk_id': 2,
                        'chunk_index': 2,
                        'text': 'beta',
                        'start_page_index': 2,
                        'end_page_index': 2,
                        'start_page_number': 3,
                        'end_page_number': 3,
                        'covered_page_indices': [2],
                        'covered_page_numbers': [3],
                    },
                ], f, ensure_ascii=False, indent=2)

            builder = build_context_builder(OmegaConf.create({
                'baselines': {
                    'name': 'colbertv2',
                    'params': {'top_k': 1, 'chunk_size': 8, 'chunk_overlap': 0},
                    'checkpoint': 'dummy-checkpoint',
                },
                'benchmarks': {
                    'name': 'mmlongbench',
                    'tmp_dir': tmp_dir,
                    'ocr_json_dir': os.path.join(tmp_dir, 'pdf_jsons'),
                    'max_pages': 3,
                }
            }))

            page_dir = os.path.join(tmp_dir, 'pdf_jsons', 'sample')
            os.makedirs(page_dir)
            for page_no, text in ((1, 'page one'), (2, 'page two'), (3, 'page three')):
                with open(os.path.join(page_dir, f'page_{page_no:04d}.json'), 'w', encoding='utf-8') as f:
                    json.dump({'text': text}, f)

            messages = builder.build(
                'mmlongbench',
                {'doc_id': 'sample.pdf', 'question_id': 'q1', 'question': 'needle?'},
            )

        self.assertEqual(messages.metadata['retrieved_chunks'][0]['chunk_id'], 1)
        self.assertEqual(messages.metadata['retrieved_pages'][0]['page_number'], 2)
        self.assertIn('needle evidence', messages[0]['content'][0]['text'])

    def _m3docrag_iterate_query_cfg(self, tmp_dir, max_pages=3):
        cfg = self._m3docrag_iterate_cfg(tmp_dir, max_pages=max_pages)
        cfg.baselines.name = 'm3docrag-iterate-query'
        cfg.baselines.online_colpali = {
            'backbone_path': '/test/backbone',
            'adapter_path': '/test/adapter',
            'device': 'cpu',
            'query_batch_size': 1,
        }
        return cfg

    def _build_m3docrag_iterate_query_mmlongbench(self, evaluator_responses, online_query_embeddings=None):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_embeddings(
                tmp_dir,
                benchmark_name='mmlongbench',
                doc_stem='sample',
                question_id='q1',
                doc_embeddings=torch.tensor([
                    [[1.0, 0.0]],
                    [[0.0, 5.0]],
                    [[3.0, 0.0]],
                ]),
                query_embedding=torch.tensor([[1.0, 0.0]]),
            )
            page_dir = os.path.join(tmp_dir, 'pdf_pngs', 'sample')
            os.makedirs(page_dir)
            for page_no in (1, 2, 3):
                Image.new('RGB', (1, 1), color='white').save(
                    os.path.join(page_dir, f'page_{page_no:04d}_dpi144.png')
                )
            builder = build_context_builder(self._m3docrag_iterate_query_cfg(tmp_dir))
            responses = list(evaluator_responses)
            query_embeddings = list(online_query_embeddings or [])
            original_call_llm_messages = m3docrag_iterate_query.call_llm_messages
            try:
                def fake_call_llm_messages(*args, **kwargs):
                    return responses.pop(0)

                def fake_encode_search_query(search_query):
                    if not query_embeddings:
                        raise AssertionError(f'unexpected online ColPali query encoding: {search_query}')
                    return query_embeddings.pop(0)

                m3docrag_iterate_query.call_llm_messages = fake_call_llm_messages
                builder._encode_search_query = fake_encode_search_query
                return builder.build(
                    'mmlongbench',
                    {'doc_id': 'sample.pdf', 'question_id': 'q1', 'question': 'Question?'},
                    client=object(),
                )
            finally:
                m3docrag_iterate_query.call_llm_messages = original_call_llm_messages

    def _m3docrag_iterate_cfg(self, tmp_dir, max_pages=3):
        return OmegaConf.create({
            'baselines': {
                'name': 'm3docrag-iterate',
                'params': {'max_iterations': 3, 'evaluator_model_name': 'eval-model'},
            },
            'benchmarks': {
                'name': 'mmlongbench',
                'tmp_dir': tmp_dir,
                'pdf_png_dir': os.path.join(tmp_dir, 'pdf_pngs'),
                'image_prefix': os.path.join(tmp_dir, 'longdoc_images'),
                'max_pages': max_pages,
                'resolution': 144,
            },
        })

    def _build_m3docrag_iterate_mmlongbench(self, evaluator_responses):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_embeddings(
                tmp_dir,
                benchmark_name='mmlongbench',
                doc_stem='sample',
                question_id='q1',
                doc_embeddings=torch.tensor([
                    [[1.0, 0.0]],
                    [[2.0, 0.0]],
                    [[3.0, 0.0]],
                ]),
                query_embedding=torch.tensor([[1.0, 0.0]]),
            )
            page_dir = os.path.join(tmp_dir, 'pdf_pngs', 'sample')
            os.makedirs(page_dir)
            for page_no in (1, 2, 3):
                Image.new('RGB', (1, 1), color='white').save(
                    os.path.join(page_dir, f'page_{page_no:04d}_dpi144.png')
                )
            builder = build_context_builder(self._m3docrag_iterate_cfg(tmp_dir))
            responses = list(evaluator_responses)
            original_call_llm_messages = m3docrag_iterate.call_llm_messages
            try:
                def fake_call_llm_messages(*args, **kwargs):
                    return responses.pop(0)

                m3docrag_iterate.call_llm_messages = fake_call_llm_messages
                return builder.build(
                    'mmlongbench',
                    {'doc_id': 'sample.pdf', 'question_id': 'q1', 'question': 'Question?'},
                    client=object(),
                )
            finally:
                m3docrag_iterate.call_llm_messages = original_call_llm_messages

    def _m3docrag_cfg(self, tmp_dir, max_pages=3):
        return OmegaConf.create({
            'baselines': {
                'name': 'm3docrag',
                'params': {'top_k': 1},
            },
            'benchmarks': {
                'name': 'mmlongbench',
                'tmp_dir': tmp_dir,
                'pdf_png_dir': os.path.join(tmp_dir, 'pdf_pngs'),
                'image_prefix': os.path.join(tmp_dir, 'longdoc_images'),
                'max_pages': max_pages,
                'resolution': 144,
            },
        })

    def _evisrag_cfg(self, tmp_dir, max_pages=3):
        return OmegaConf.create({
            'baselines': {
                'name': 'evisrag',
                'params': {'top_k': 3, 'max_images': 3},
            },
            'benchmarks': {
                'name': 'mmlongbench',
                'tmp_dir': tmp_dir,
                'pdf_png_dir': os.path.join(tmp_dir, 'pdf_pngs'),
                'image_prefix': os.path.join(tmp_dir, 'longdoc_images'),
                'max_pages': max_pages,
                'resolution': 144,
            },
        })

    def _write_visrag_embeddings(self, tmp_dir, benchmark_name, doc_stem, question_id, doc_embeddings, query_embedding):
        pdf_path = visrag_pdf_embeddings_path(benchmark_name, doc_stem)
        question_path = visrag_question_embeddings_path(benchmark_name, question_id)
        os.makedirs(pdf_path.parent, exist_ok=True)
        os.makedirs(question_path.parent, exist_ok=True)
        save_file({'embeddings': doc_embeddings}, pdf_path)
        save_file({'query_embedding': query_embedding}, question_path)

    def _write_embeddings(self, tmp_dir, benchmark_name, doc_stem, question_id, doc_embeddings, query_embedding):
        pdf_path = colpali_pdf_embeddings_path(benchmark_name, doc_stem)
        question_path = colpali_question_embeddings_path(benchmark_name, question_id)
        os.makedirs(pdf_path.parent, exist_ok=True)
        os.makedirs(question_path.parent, exist_ok=True)
        save_file({'embeddings': doc_embeddings}, pdf_path)
        save_file({'query_embedding': query_embedding}, question_path)


if __name__ == '__main__':
    unittest.main()
