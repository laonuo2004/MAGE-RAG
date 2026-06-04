import json
import logging
import os

from baselines.base import ContextBuilder, ContextMessages
from baselines.image import VISION_SYSTEM_PROMPT
from baselines.m3docrag import m3docragContextBuilder
from baselines.utils.benchmarks_related import (
    allowed_page_indices,
    encode_image_file_to_base64,
    encode_pil_image_to_base64,
)
from benchmarks.utils.data_utils import (
    colpali_pdf_embeddings_path,
    colpali_question_embeddings_path,
    mmlongbench_file_id,
    mmlongbench_png_page_path,
)
from utils.config_utils import require_config_value
from utils.llm_utils import call_llm_messages, completion_content

logger = logging.getLogger(__name__)


class M3DocRAGIterateContextBuilder(ContextBuilder):
    name = 'm3docrag-iterate'
    _load_embeddings = m3docragContextBuilder._load_embeddings
    _retrieve_pages = m3docragContextBuilder._retrieve_pages
    _longdocurl_image_path = m3docragContextBuilder._longdocurl_image_path

    def __init__(self, cfg=None):
        super().__init__(cfg)
        max_iterations = require_config_value(self.cfg, 'baselines.params.max_iterations')
        self.max_iterations = int(max_iterations)
        if self.max_iterations <= 0:
            raise ValueError('M3DocRAG iterate baseline requires cfg.baselines.params.max_iterations > 0.')
        self.top_k = self.max_iterations
        self.evaluator_model_name = str(require_config_value(self.cfg, 'baselines.params.evaluator_model_name'))

    def build_mmlongbench(self, sample, **kwargs):
        doc_id = sample['doc_id']
        question_id = sample['question_id']
        pdf_embedding_path = colpali_pdf_embeddings_path('mmlongbench', mmlongbench_file_id(doc_id))
        question_embedding_path = colpali_question_embeddings_path('mmlongbench', question_id)
        doc_embeddings, query_embedding = self._load_embeddings(pdf_embedding_path, question_embedding_path)
        benchmark_cfg = require_config_value(self.cfg, 'benchmarks')
        allowed_pages = allowed_page_indices('mmlongbench', sample, benchmark_cfg, doc_embeddings.shape[0])
        candidate_pages = self._retrieve_pages(doc_embeddings, query_embedding, allowed_pages)
        selected_pages, iteration_trace, stopped_by = self._select_pages(
            sample,
            candidate_pages,
            'mmlongbench',
            kwargs.get('client'),
        )

        content = self._mmlongbench_reader_content(sample, selected_pages)
        return ContextMessages(
            [{'role': 'user', 'content': content}],
            metadata=self._metadata(
                candidate_pages,
                selected_pages,
                iteration_trace,
                stopped_by,
                allowed_pages,
                pdf_embedding_path,
                question_embedding_path,
            ),
        )

    def build_longdocurl(self, sample, **kwargs):
        doc_no = sample['doc_no']
        question_id = sample['question_id']
        pdf_embedding_path = colpali_pdf_embeddings_path('longdocurl', doc_no)
        question_embedding_path = colpali_question_embeddings_path('longdocurl', question_id)
        doc_embeddings, query_embedding = self._load_embeddings(pdf_embedding_path, question_embedding_path)
        benchmark_cfg = require_config_value(self.cfg, 'benchmarks')
        allowed_pages = allowed_page_indices('longdocurl', sample, benchmark_cfg, doc_embeddings.shape[0])
        candidate_pages = self._retrieve_pages(doc_embeddings, query_embedding, allowed_pages)
        selected_pages, iteration_trace, stopped_by = self._select_pages(
            sample,
            candidate_pages,
            'longdocurl',
            kwargs.get('client'),
        )

        content = self._longdocurl_reader_content(sample, selected_pages)
        return ContextMessages(
            [{'role': 'user', 'content': content}],
            metadata=self._metadata(
                candidate_pages,
                selected_pages,
                iteration_trace,
                stopped_by,
                allowed_pages,
                pdf_embedding_path,
                question_embedding_path,
            ),
        )

    def _select_pages(self, sample, candidate_pages, benchmark_name, client):
        iteration_trace = []
        if client is None:
            return candidate_pages, iteration_trace, 'fallback_no_client'

        fallback_reason = 'fallback_max_iterations'
        for iteration in range(1, len(candidate_pages) + 1):
            pages = candidate_pages[:iteration]
            trace = {
                'iteration': iteration,
                'page_numbers': [page['page_number'] for page in pages],
                'page_indices': [page['page_index'] for page in pages],
            }
            try:
                response = self._call_evaluator(sample, pages, benchmark_name, client)
                trace['raw_response'] = response
                verdict = self._parse_evaluator_response(response)
                trace.update(verdict)
            except Exception as exc:
                trace['error'] = str(exc)
                iteration_trace.append(trace)
                return candidate_pages, iteration_trace, 'fallback_evaluator_error'

            iteration_trace.append(trace)
            if verdict['answerable']:
                return pages, iteration_trace, 'answerable'

        return candidate_pages, iteration_trace, fallback_reason

    def _call_evaluator(self, sample, pages, benchmark_name, client):
        messages = [{'role': 'user', 'content': self._evaluator_content(sample, pages, benchmark_name)}]
        completion = call_llm_messages(
            client,
            self.evaluator_model_name,
            messages,
            max_tokens=4096,
            temperature=0.0,
            retries=1,
            logger=logger,
            log_prefix='M3DocRAG iterate evaluator',
            failure_value=lambda exc: f'Failed: {exc}',
        )
        return completion_content(completion)

    def _parse_evaluator_response(self, response):
        payload = json.loads(str(response))
        if not isinstance(payload, dict):
            raise ValueError('Evaluator response must be a JSON object.')
        answerable = payload.get('answerable')
        if not isinstance(answerable, bool):
            raise ValueError('Evaluator response field "answerable" must be boolean.')
        return {
            'answerable': answerable,
            'reason': str(payload.get('reason') or ''),
            'missing_evidence': str(payload.get('missing_evidence') or ''),
        }

    def _evaluator_content(self, sample, pages, benchmark_name):
        prompt = (
            'You are judging whether the provided document page images contain enough evidence '
            'to answer the question. Use only the images. Return strict JSON only with this schema: '
            '{"answerable": true/false, "reason": "...", "missing_evidence": "..."}\n'
            f'Question: {sample["question"]}'
        )
        content = [{'type': 'text', 'text': prompt}]
        for index, page in enumerate(pages):
            content.append({
                'type': 'text',
                'text': f'Page candidate {index + 1}: document page {page["page_number"]}.\n',
            })
            content.append(
                self._mmlongbench_image_part(sample, page)
                if benchmark_name == 'mmlongbench'
                else self._longdocurl_image_part(sample, page)
            )
        return content

    def _mmlongbench_reader_content(self, sample, pages):
        content = [{'type': 'text', 'text': sample['question']}]
        for page in pages:
            content.append(self._mmlongbench_image_part(sample, page))
        return content

    def _longdocurl_reader_content(self, sample, pages):
        prompt = (
            VISION_SYSTEM_PROMPT
            + 'Following is our question: \n'
            + f'<question>{sample["question"]}</question>'
            + '\n'
        )
        content = [{'type': 'text', 'text': prompt}]
        for index, page in enumerate(pages):
            content.extend([
                {'type': 'text', 'text': f'Below is the {index + 1}-th image (total {len(pages)} images).\n'},
                self._longdocurl_image_part(sample, page),
            ])
        return content

    def _mmlongbench_image_part(self, sample, page):
        from PIL import Image

        benchmark_cfg = require_config_value(self.cfg, 'benchmarks')
        page_path = mmlongbench_png_page_path(benchmark_cfg, sample['doc_id'], page['page_index'])
        if not os.path.exists(page_path):
            raise FileNotFoundError(f'Missing preprocessed PNG page for doc_id={sample["doc_id"]}: {page_path}')
        with Image.open(page_path) as image:
            encoded_image = encode_pil_image_to_base64(image)
        return {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{encoded_image}'}}

    def _longdocurl_image_part(self, sample, page):
        image_path = self._longdocurl_image_path(sample['doc_no'], page['page_index'])
        return {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{encode_image_file_to_base64(image_path)}'}}

    def _metadata(
        self,
        candidate_pages,
        retrieved_pages,
        iteration_trace,
        stopped_by,
        allowed_pages,
        pdf_embedding_path,
        question_embedding_path,
    ):
        return {
            'context_builder': self.name,
            'candidate_pages': candidate_pages,
            'retrieved_pages': retrieved_pages,
            'allowed_pages': list(allowed_pages),
            'iteration_trace': iteration_trace,
            'stopped_by': stopped_by,
            'max_iterations': self.max_iterations,
            'evaluator_model_name': self.evaluator_model_name,
            'embedding_paths': {
                'pdf': str(pdf_embedding_path),
                'question': str(question_embedding_path),
            },
        }

