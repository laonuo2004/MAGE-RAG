import unittest
import base64
import json
import os
import tempfile
from io import BytesIO

from omegaconf import OmegaConf

from baselines.wrapper import build_context_builder
from benchmarks import wrapper


class ContextBuilderTests(unittest.TestCase):
    def test_build_context_builder_selects_image_and_ocr(self):
        image_builder = build_context_builder(OmegaConf.create({'baselines': {'name': 'image'}}))
        ocr_builder = build_context_builder(OmegaConf.create({'baselines': {'name': 'ocr'}}))

        self.assertEqual(image_builder.name, 'image')
        self.assertEqual(ocr_builder.name, 'ocr')

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
        self.assertEqual(messages.metadata, {'context_builder': 'image'})

    def test_wrapper_routes_longdocurl_ocr_baseline(self):
        captured = []
        original_run_longdocurl = wrapper._run_longdocurl
        try:
            wrapper._run_longdocurl = lambda cfg: captured.append(cfg)
            wrapper.run_benchmark(OmegaConf.create({
                'benchmarks': {'name': 'longdocurl', 'qa_file': 'qa.jsonl', 'model_name': 'model'},
                'baselines': {'name': 'ocr'},
            }))
        finally:
            wrapper._run_longdocurl = original_run_longdocurl

        self.assertEqual(captured[0].baselines.name, 'ocr')

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

        prompt = messages[0]['content']
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


if __name__ == '__main__':
    unittest.main()
