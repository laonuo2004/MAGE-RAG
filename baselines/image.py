import os

from .base import ContextBuilder, ContextMessages
from baselines.utils.baselines_utils import encode_image_file_to_base64, encode_pil_image_to_base64
from benchmarks.mmlongbench.utils.preprocess_cache import mmlongbench_png_page_path


VISION_SYSTEM_PROMPT = 'You are an expert in visual document question-answering, please answer our questions based on the given images.\n'


class ImageContextBuilder(ContextBuilder):
    name = 'image'

    def build_mmlongbench(self, sample, **kwargs):
        from PIL import Image

        question = sample['question']
        image_list = []
        for index in range(int(self.cfg.benchmarks.max_pages)):
            page_path = mmlongbench_png_page_path(self.cfg.benchmarks, sample['doc_id'], index)
            if not os.path.exists(page_path):
                if index == 0:
                    raise FileNotFoundError(
                        f'Missing preprocessed PNG cache for doc_id={sample["doc_id"]}: {page_path}. '
                        'Run benchmarks/mmlongbench/scripts/preprocess_mmlongbench.py before evaluating the image baseline.'
                    )
                break
            image = Image.open(page_path)
            image_list.append(encode_pil_image_to_base64(image))

        content = [{'type': 'text', 'text': question}]
        for encoded_image in image_list:
            content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{encoded_image}'}})
        return ContextMessages(
            [{'role': 'user', 'content': content}],
            metadata={'context_builder': self.name},
        )

    def build_longdocurl(self, sample, **kwargs):
        question = sample['question']
        prompt = VISION_SYSTEM_PROMPT + 'Following is our question: \n' + f'<question>{question}</question>' + '\n'
        content = [{'type': 'text', 'text': prompt}]
        images = sample.get('images')
        if isinstance(images, str):
            images = [images]
        for index, image_path in enumerate(images or []):
            content.extend([
                {'type': 'text', 'text': f'Below is the {index + 1}-th image (total {len(images)} images).\n'},
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{encode_image_file_to_base64(image_path)}'}},
            ])
        return ContextMessages(
            [{'role': 'user', 'content': content}],
            metadata={'context_builder': self.name},
        )
