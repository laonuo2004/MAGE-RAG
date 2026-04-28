import base64
import os
import re
from io import BytesIO
from threading import Lock

from .base import ContextBuilder, ContextMessages


_doc_locks = {}
_doc_locks_guard = Lock()


VISION_SYSTEM_PROMPT = 'You are an expert in visual document question-answering, please answer our questions based on the given images.\n'


def _get_doc_lock(doc_id):
    with _doc_locks_guard:
        if doc_id not in _doc_locks:
            _doc_locks[doc_id] = Lock()
        return _doc_locks[doc_id]


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

    def _render_page_path(self, sample, page_index):
        doc_name = re.sub(r'\.pdf$', '', sample['doc_id']).split('/')[-1]
        tmp_dir = self.cfg.benchmarks.tmp_dir
        os.makedirs(tmp_dir, exist_ok=True)
        return os.path.join(tmp_dir, f'{doc_name}_{page_index + 1}.png')

    def _build_mmlongbench_openai(self, sample):
        import fitz
        from PIL import Image

        question = sample['question']
        image_list = []
        with _get_doc_lock(sample['doc_id']):
            pdf_path = os.path.join(self.cfg.benchmarks.document_path, sample['doc_id'])
            max_pages = self.cfg.benchmarks.max_pages
            with fitz.open(pdf_path) as pdf:
                for index, page in enumerate(pdf[:max_pages]):
                    page_path = self._render_page_path(sample, index)
                    if not os.path.exists(page_path):
                        image = page.get_pixmap(dpi=self.cfg.benchmarks.resolution)
                        image.save(page_path)
                    image = Image.open(page_path)
                    image_list.append(_encode_pil_image_to_base64(image))

        content = [{'type': 'text', 'text': question}]
        for encoded_image in image_list:
            content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{encoded_image}'}})
        return ContextMessages(
            [{'role': 'user', 'content': content}],
            metadata={'context_builder': self.name, 'input_format': 'image'},
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
            metadata={'context_builder': self.name, 'sample': sample},
        )
