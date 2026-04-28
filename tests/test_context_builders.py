import unittest
from types import SimpleNamespace

from baselines.wrapper import build_context_builder
from benchmarks import wrapper


class ContextBuilderTests(unittest.TestCase):
    def test_build_context_builder_selects_image_and_ocr(self):
        image_builder = build_context_builder(SimpleNamespace(baselines={'name': 'image'}))
        ocr_builder = build_context_builder(SimpleNamespace(baselines={'name': 'ocr'}))

        self.assertEqual(image_builder.name, 'image')
        self.assertEqual(ocr_builder.name, 'ocr')

    def test_build_context_builder_supports_baseline_name(self):
        image_builder = build_context_builder(SimpleNamespace(baselines=SimpleNamespace(name='image')))
        ocr_builder = build_context_builder({'name': 'ocr'})

        self.assertEqual(image_builder.name, 'image')
        self.assertEqual(ocr_builder.name, 'ocr')

    def test_longdocurl_image_context_matches_legacy_prompt_shape(self):
        sample = {'question': 'What is shown?', 'images': ['page1.png', 'page2.png']}
        builder = build_context_builder({'name': 'image'})

        bundle = builder.build('longdocurl', sample, SimpleNamespace())

        self.assertEqual(
            bundle.prompt,
            'You are an expert in visual document question-answering, please answer our questions based on the given images.\n'
            'Following is our question: \n'
            '<question>What is shown?</question>\n',
        )
        self.assertEqual(bundle.images, ['page1.png', 'page2.png'])
        self.assertEqual(
            bundle.system_prompt,
            'You are an expert in visual document question-answering, please answer our questions based on the given images.\n',
        )
        self.assertEqual(bundle.metadata, {'context_builder': 'image', 'input_format': 'e2e'})

    def test_wrapper_maps_longdocurl_ocr_builder_to_ocr_input_format(self):
        captured = []
        original_run_longdocurl = wrapper._run_longdocurl
        try:
            wrapper._run_longdocurl = lambda cfg, benchmark_cfg, context_builder_name: captured.append(
                (benchmark_cfg, context_builder_name)
            )
            wrapper.run_benchmark({
                'benchmarks': {'name': 'longdocurl', 'qa_file': 'qa.jsonl', 'model_name': 'model'},
                'baselines': {'name': 'ocr'},
                'llm_providers': {'model_mapping': {}},
            })
        finally:
            wrapper._run_longdocurl = original_run_longdocurl

        self.assertEqual(captured[0][1], 'ocr')


if __name__ == '__main__':
    unittest.main()
