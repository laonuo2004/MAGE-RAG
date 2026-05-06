import os
import json

from .base import ContextBuilder, ContextMessages
from benchmarks.mmlongbench.utils.preprocess_cache import mmlongbench_ocr_page_path
from utils.config_utils import require_config_value

TEXT_SYSTEM_PROMPT = 'You are an expert in document question-answering, please answer our questions based on the extracted text from the given pages.\n'


class OcrContextBuilder(ContextBuilder):
    name = 'ocr'

    def build_mmlongbench(self, sample, **kwargs):
        question = sample['question']
        benchmark_cfg = require_config_value(self.cfg, 'benchmarks')

        page_blocks = []
        for page_index in range(int(require_config_value(benchmark_cfg, 'max_pages'))):
            page_path = mmlongbench_ocr_page_path(benchmark_cfg, sample['doc_id'], page_index)
            if not os.path.exists(page_path):
                if page_index == 0:
                    raise FileNotFoundError(
                        f'Missing preprocessed OCR cache for doc_id={sample["doc_id"]}: {page_path}. '
                        'Run benchmarks/mmlongbench/scripts/preprocess_mmlongbench.py before evaluating the ocr baseline.'
                    )
                break
            with open(page_path, 'r', encoding='utf-8') as f:
                page_payload = json.load(f)
            page_text = str(page_payload.get('text') or '').strip() or '[EMPTY PAGE]'
            page_blocks.append(f'[Page {page_index + 1}]\n{page_text}')

        document_text = '\n\n'.join(page_blocks)
        prompt = (
            'You are given the OCR/text extracted from a long PDF document.\n'
            'Answer the question using only the provided document text.\n'
            'If the answer cannot be found, say Not answerable.\n\n'
            f'Question:\n{question}\n\n'
            f'Document text:\n{document_text}'
        )
        return ContextMessages(
            [{'role': 'user', 'content': prompt}],
            metadata={'context_builder': self.name},
        )

    def build_longdocurl(self, sample, **kwargs):
        from benchmarks.longdocurl.eval.api_models.pure_ocr_utils import get_pure_ocr_prompt_pymupdf

        question = sample['question']
        benchmark_cfg = require_config_value(self.cfg, 'benchmarks')
        ocr_prompt, ocr_pages_used = get_pure_ocr_prompt_pymupdf(
            sample['doc_no'],
            images=sample.get('images'),
            ocr_json_dir=require_config_value(benchmark_cfg, 'ocr_json_dir'),
            start_page=sample['start_end_idx'][0],
            end_page=sample['start_end_idx'][1],
        )
        prompt = (
            TEXT_SYSTEM_PROMPT
            + 'Following is our question: \n'
            + f'<question>{question}</question>\n'
            + 'Following are the extracted texts from the selected document pages:\n'
            + ocr_prompt
        )
        return ContextMessages(
            [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}],
            metadata={'context_builder': self.name},
        )
