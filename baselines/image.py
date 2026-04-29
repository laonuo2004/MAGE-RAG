import base64
import os
from io import BytesIO

from .base import ContextBuilder, ContextMessages
from benchmarks.mmlongbench.utils.preprocess_cache import mmlongbench_png_page_path


VISION_SYSTEM_PROMPT = 'You are an expert in visual document question-answering, please answer our questions based on the given images.\n'


def _encode_pil_image_to_base64(img):
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def _encode_image_file_to_base64(image_path):
    if 'https' in image_path:
        import requests

        response = requests.get(image_path)
        return base64.b64encode(response.content).decode('utf-8')
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


class ImageContextBuilder(ContextBuilder):
    name = 'image'

    def build_mmlongbench(self, sample, **kwargs):
        return self._build_mmlongbench_openai(sample)

    def _build_mmlongbench_openai(self, sample):
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
            image_list.append(_encode_pil_image_to_base64(image))

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
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{_encode_image_file_to_base64(image_path)}'}},
            ])
        return ContextMessages(
            [{'role': 'user', 'content': content}],
            metadata={'context_builder': self.name},
        )
