import base64
import os
import re
from io import BytesIO
from threading import Lock

from .base import ContextBuilder, ContextBundle


_cached_gemini_images = {}
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


class ImageContextBuilder(ContextBuilder):
    name = 'image'

    def build_mmlongbench(self, sample, args, mode='png', **kwargs):
        if args.api_style == 'openai':
            return self._build_mmlongbench_openai(sample, args)
        if args.api_style == 'gemini':
            return self._build_mmlongbench_gemini(sample, args, mode=mode)
        raise AssertionError()

    def _render_page_path(self, sample, args, page_index):
        doc_name = re.sub(r'\.pdf$', '', sample['doc_id']).split('/')[-1]
        tmp_dir = getattr(args, 'tmp_dir', './tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        return os.path.join(tmp_dir, f'{doc_name}_{page_index + 1}.png')

    def _build_mmlongbench_openai(self, sample, args):
        import fitz
        from PIL import Image

        question = sample['question']
        image_list = []
        with _get_doc_lock(sample['doc_id']):
            with fitz.open(os.path.join(args.document_path, sample['doc_id'])) as pdf:
                for index, page in enumerate(pdf[: args.max_pages]):
                    page_path = self._render_page_path(sample, args, index)
                    if not os.path.exists(page_path):
                        image = page.get_pixmap(dpi=args.resolution)
                        image.save(page_path)
                    image = Image.open(page_path)
                    image_list.append(_encode_pil_image_to_base64(image))

        content = [{'type': 'text', 'text': question}]
        for encoded_image in image_list:
            content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{encoded_image}'}})
        return ContextBundle(
            messages=[{'role': 'user', 'content': content}],
            metadata={'context_builder': self.name, 'input_format': 'image'},
        )

    def _build_mmlongbench_gemini(self, sample, args, mode='png'):
        import fitz
        from PIL import Image
        import google.generativeai as genai

        question = sample['question']
        image_list = []
        with fitz.open(os.path.join(args.document_path, sample['doc_id'])) as pdf:
            if mode == 'png':
                for index, page in enumerate(pdf[: args.max_pages]):
                    page_path = self._render_page_path(sample, args, index)
                    if not os.path.exists(page_path):
                        image = page.get_pixmap(dpi=args.resolution)
                        image.save(page_path)
                    image_list.append(Image.open(page_path))
            else:
                if sample['doc_id'] in _cached_gemini_images:
                    image_list = _cached_gemini_images[sample['doc_id']]
                else:
                    for index, page in enumerate(pdf[: args.max_pages]):
                        page_path = self._render_page_path(sample, args, index)
                        if not os.path.exists(page_path):
                            image = page.get_pixmap(dpi=args.resolution)
                            image.save(page_path)
                        image_list.append(genai.upload_file(page_path))
                    _cached_gemini_images[sample['doc_id']] = image_list
        return ContextBundle(
            messages=[question] + image_list,
            metadata={'context_builder': self.name, 'input_format': 'image', 'gemini_image_mode': mode},
        )

    def build_longdocurl(self, sample, args, **kwargs):
        question = sample['question']
        prompt = VISION_SYSTEM_PROMPT + 'Following is our question: \n' + f'<question>{question}</question>' + '\n'
        return ContextBundle(
            prompt=prompt,
            images=sample['images'],
            system_prompt=VISION_SYSTEM_PROMPT,
            metadata={'context_builder': self.name, 'input_format': 'e2e'},
        )
