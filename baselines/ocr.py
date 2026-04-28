import os

from .base import ContextBuilder, ContextBundle


TEXT_SYSTEM_PROMPT = 'You are an expert in document question-answering, please answer our questions based on the extracted text from the given pages.\n'


class OcrContextBuilder(ContextBuilder):
    name = 'ocr'

    def build_mmlongbench(self, sample, args, **kwargs):
        import fitz

        question = sample['question']
        pdf_path = os.path.join(args.document_path, sample['doc_id'])

        page_blocks = []
        with fitz.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf[: args.max_pages], start=1):
                page_text = page.get_text('text').strip()
                if not page_text:
                    page_text = '[EMPTY PAGE]'
                page_blocks.append(f'[Page {page_idx}]\n{page_text}')

        document_text = '\n\n'.join(page_blocks)
        prompt = (
            'You are given the OCR/text extracted from a long PDF document.\n'
            'Answer the question using only the provided document text.\n'
            'If the answer cannot be found, say Not answerable.\n\n'
            f'Question:\n{question}\n\n'
            f'Document text:\n{document_text}'
        )
        return ContextBundle(
            messages=[{'role': 'user', 'content': prompt}],
            metadata={'context_builder': self.name, 'input_format': 'ocr'},
        )

    def build_longdocurl(self, sample, args, **kwargs):
        from pure_ocr_utils import get_pure_ocr_prompt_pymupdf

        question = sample['question']
        ocr_prompt, ocr_pages_used = get_pure_ocr_prompt_pymupdf(
            sample['doc_no'],
            images=sample.get('images'),
            ocr_json_dir=args.ocr_json_dir,
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
        return ContextBundle(
            prompt=prompt,
            images=None,
            system_prompt=TEXT_SYSTEM_PROMPT,
            metadata={'context_builder': self.name, 'input_format': 'ocr', 'ocr_backend': 'pymupdf', 'ocr_pages_used': ocr_pages_used},
        )
