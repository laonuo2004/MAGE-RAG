import os

from .base import ContextBuilder, ContextMessages, build_context_summary, build_logical_cost, build_retrieval_metadata
from benchmarks.utils.document_preprocess import encode_image_file_to_base64, encode_pil_image_to_base64
from benchmarks.utils.data_utils import mmlongbench_png_page_path
from utils.config_utils import require_config_value


VISION_SYSTEM_PROMPT = 'You are an expert in visual document question-answering, please answer our questions based on the given images.\n'


class ImageContextBuilder(ContextBuilder):
    name = 'image'

    def build_mmlongbench(self, sample, **kwargs):
        from PIL import Image

        question = sample['question']
        benchmark_cfg = require_config_value(self.cfg, 'benchmarks')
        image_list = []
        for index in range(int(require_config_value(benchmark_cfg, 'max_pages'))):
            page_path = mmlongbench_png_page_path(benchmark_cfg, sample['doc_id'], index)
            if not os.path.exists(page_path):
                if index == 0:
                    raise FileNotFoundError(
                        f'Missing preprocessed PNG cache for doc_id={sample["doc_id"]}: {page_path}. '
                        'Run benchmarks/scripts/preprocess_documents.py --benchmark mmlongbench --mode image first.'
                    )
                break
            image = Image.open(page_path)
            image_list.append(encode_pil_image_to_base64(image))

        content = [{'type': 'text', 'text': question}]
        for encoded_image in image_list:
            content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{encoded_image}'}})
        page_ids = list(range(len(image_list)))
        return ContextMessages(
            [{'role': 'user', 'content': content}],
            metadata=self._metadata(page_ids, len(image_list), question),
        )

    def build_longdocurl(self, sample, **kwargs):
        question = sample['question']
        prompt = VISION_SYSTEM_PROMPT + 'Following is our question: \n' + f'<question>{question}</question>' + '\n'
        content = [{'type': 'text', 'text': prompt}]
        images = sample.get('images')
        if isinstance(images, str):
            images = [images]
        images = images or []
        for index, image_path in enumerate(images or []):
            content.extend([
                {'type': 'text', 'text': f'Below is the {index + 1}-th image (total {len(images)} images).\n'},
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{encode_image_file_to_base64(image_path)}'}},
            ])
        page_ids = list(range(len(images)))
        return ContextMessages(
            [{'role': 'user', 'content': content}],
            metadata=self._metadata(page_ids, len(images), prompt),
        )

    def _metadata(self, page_ids, image_count, text_prompt):
        return {
            'context_builder': self.name,
            'retrieval': build_retrieval_metadata(
                retrieved_items=[
                    {'rank': index + 1, 'page_index': page_id, 'page_number': page_id + 1}
                    for index, page_id in enumerate(page_ids)
                ],
                initial_retrieved_pages=list(page_ids),
                final_context_pages=list(page_ids),
            ),
            'context_summary': build_context_summary(
                page_ids=list(page_ids),
                num_context_pages=len(page_ids),
                num_image_units=image_count,
                num_text_chars=len(str(text_prompt or '')),
            ),
            'logical_cost': build_logical_cost(
                num_input_text_chars=len(str(text_prompt or '')),
                num_input_images=image_count,
                num_context_pages=len(page_ids),
                num_final_evidence_units=image_count,
            ),
        }
