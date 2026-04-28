import unittest
import base64
import os
import tempfile

from baselines.wrapper import build_context_builder
from benchmarks import wrapper


class ContextBuilderTests(unittest.TestCase):
    def test_build_context_builder_selects_image_and_ocr(self):
        image_builder = build_context_builder({'baselines': {'name': 'image'}})
        ocr_builder = build_context_builder({'baselines': {'name': 'ocr'}})

        self.assertEqual(image_builder.name, 'image')
        self.assertEqual(ocr_builder.name, 'ocr')

    def test_build_context_builder_supports_direct_baseline_cfg(self):
        image_builder = build_context_builder({'name': 'image'})
        ocr_builder = build_context_builder({'name': 'ocr'})

        self.assertEqual(image_builder.name, 'image')
        self.assertEqual(ocr_builder.name, 'ocr')

    def test_longdocurl_image_context_matches_legacy_prompt_shape(self):
        png_bytes = base64.b64decode(
            'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII='
        )
        builder = build_context_builder({'baselines': {'name': 'image'}, 'benchmarks': {'name': 'longdocurl'}})

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
        self.assertEqual(messages.metadata['input_format'], 'e2e')
        self.assertEqual(
            messages.metadata['system_prompt'],
            'You are an expert in visual document question-answering, please answer our questions based on the given images.\n',
        )

    def test_wrapper_maps_longdocurl_ocr_builder_to_ocr_input_format(self):
        captured = []
        original_run_longdocurl = wrapper._run_longdocurl
        try:
            wrapper._run_longdocurl = lambda cfg: captured.append(cfg)
            wrapper.run_benchmark({
                'benchmarks': {'name': 'longdocurl', 'qa_file': 'qa.jsonl', 'model_name': 'model'},
                'baselines': {'name': 'ocr'},
                'llm_providers': {'model_mapping': {}},
            })
        finally:
            wrapper._run_longdocurl = original_run_longdocurl

        self.assertEqual(captured[0]['baselines']['name'], 'ocr')


if __name__ == '__main__':
    unittest.main()
