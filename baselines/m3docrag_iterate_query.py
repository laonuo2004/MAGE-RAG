import contextlib
import gc
import json
import logging
import os
import sys
from pathlib import Path

import torch

from baselines.m3docrag import m3docragContextBuilder
from baselines.m3docrag_iterate import M3DocRAGIterateContextBuilder
from baselines.utils.benchmarks_related import allowed_page_indices, resolve_embedding_path
from benchmarks.mmlongbench.utils.preprocess_cache import mmlongbench_file_id
from utils.config_utils import get_config_value, require_config_value
from utils.llm_utils import call_llm_messages, completion_content

logger = logging.getLogger(__name__)

DEFAULT_BACKBONE_PATH = '/root/autodl-tmp/ylz/models/colpaligemma-3b-mix-448-base'
DEFAULT_ADAPTER_PATH = '/root/autodl-tmp/ylz/models/colpali-v1.2'


class M3DocRAGIterateQueryContextBuilder(M3DocRAGIterateContextBuilder):
    name = 'm3docrag-iterate-query'

    _load_embeddings = m3docragContextBuilder._load_embeddings
    _longdocurl_image_path = m3docragContextBuilder._longdocurl_image_path

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.top_k = 1
        self.online_colpali = {
            'backbone_path': str(get_config_value(
                self.cfg,
                'baselines.online_colpali.backbone_path',
                DEFAULT_BACKBONE_PATH,
            )),
            'adapter_path': str(get_config_value(
                self.cfg,
                'baselines.online_colpali.adapter_path',
                DEFAULT_ADAPTER_PATH,
            )),
            'device': str(get_config_value(self.cfg, 'baselines.online_colpali.device', 'auto')),
            'query_batch_size': int(get_config_value(self.cfg, 'baselines.online_colpali.query_batch_size', 1)),
        }
        if self.online_colpali['query_batch_size'] <= 0:
            raise ValueError('M3DocRAG iterate-query baseline requires online_colpali.query_batch_size > 0.')
        self._online_device = None

    def build_mmlongbench(self, sample, **kwargs):
        doc_id = sample['doc_id']
        question_id = sample['question_id']
        pdf_embedding_path = resolve_embedding_path(
            self.cfg,
            'pdf_embeddings_colpali',
            'mmlongbench',
            mmlongbench_file_id(doc_id),
        )
        question_embedding_path = resolve_embedding_path(
            self.cfg,
            'question_embeddings_colpali',
            'mmlongbench',
            question_id,
        )
        doc_embeddings, query_embedding = self._load_embeddings(pdf_embedding_path, question_embedding_path)
        benchmark_cfg = require_config_value(self.cfg, 'benchmarks')
        allowed_pages = allowed_page_indices('mmlongbench', sample, benchmark_cfg, doc_embeddings.shape[0])
        initial_pages = self._retrieve_pages(doc_embeddings, query_embedding, allowed_pages)
        selected_pages, iteration_trace, stopped_by = self._select_pages(
            sample,
            initial_pages,
            doc_embeddings,
            allowed_pages,
            'mmlongbench',
            kwargs.get('client'),
        )

        content = self._mmlongbench_reader_content(sample, selected_pages)
        return self._messages(
            content,
            initial_pages,
            selected_pages,
            iteration_trace,
            stopped_by,
            allowed_pages,
            pdf_embedding_path,
            question_embedding_path,
        )

    def build_longdocurl(self, sample, **kwargs):
        doc_no = sample['doc_no']
        question_id = sample['question_id']
        pdf_embedding_path = resolve_embedding_path(self.cfg, 'pdf_embeddings_colpali', 'longdocurl', doc_no)
        question_embedding_path = resolve_embedding_path(
            self.cfg,
            'question_embeddings_colpali',
            'longdocurl',
            question_id,
        )
        doc_embeddings, query_embedding = self._load_embeddings(pdf_embedding_path, question_embedding_path)
        benchmark_cfg = require_config_value(self.cfg, 'benchmarks')
        allowed_pages = allowed_page_indices('longdocurl', sample, benchmark_cfg, doc_embeddings.shape[0])
        initial_pages = self._retrieve_pages(doc_embeddings, query_embedding, allowed_pages)
        selected_pages, iteration_trace, stopped_by = self._select_pages(
            sample,
            initial_pages,
            doc_embeddings,
            allowed_pages,
            'longdocurl',
            kwargs.get('client'),
        )

        content = self._longdocurl_reader_content(sample, selected_pages)
        return self._messages(
            content,
            initial_pages,
            selected_pages,
            iteration_trace,
            stopped_by,
            allowed_pages,
            pdf_embedding_path,
            question_embedding_path,
        )

    def _messages(
        self,
        content,
        candidate_pages,
        retrieved_pages,
        iteration_trace,
        stopped_by,
        allowed_pages,
        pdf_embedding_path,
        question_embedding_path,
    ):
        from baselines.base import ContextMessages

        return ContextMessages(
            [{'role': 'user', 'content': content}],
            metadata=self._metadata(
                candidate_pages,
                retrieved_pages,
                iteration_trace,
                stopped_by,
                allowed_pages,
                pdf_embedding_path,
                question_embedding_path,
            ),
        )

    def _select_pages(self, sample, initial_pages, doc_embeddings, allowed_pages, benchmark_name, client):
        selected_pages = list(initial_pages[:1])
        iteration_trace = []
        if client is None:
            return selected_pages, iteration_trace, 'fallback_no_client'
        if not selected_pages:
            return selected_pages, iteration_trace, 'fallback_no_new_page'

        for iteration in range(1, self.max_iterations + 1):
            trace = {
                'iteration': iteration,
                'page_numbers': [page['page_number'] for page in selected_pages],
                'page_indices': [page['page_index'] for page in selected_pages],
            }
            try:
                response = self._call_evaluator(sample, selected_pages, benchmark_name, client)
                trace['raw_response'] = response
                verdict = self._parse_evaluator_response(response)
                trace.update(verdict)
            except Exception as exc:
                trace['error'] = str(exc)
                iteration_trace.append(trace)
                return selected_pages, iteration_trace, 'fallback_evaluator_error'

            if verdict['answerable']:
                iteration_trace.append(trace)
                return selected_pages, iteration_trace, 'answerable'

            search_query = verdict['search_query'].strip()
            if not search_query:
                iteration_trace.append(trace)
                return selected_pages, iteration_trace, 'fallback_empty_search_query'
            if len(selected_pages) >= self.max_iterations:
                iteration_trace.append(trace)
                return selected_pages, iteration_trace, 'fallback_max_iterations'

            try:
                query_embedding = self._encode_search_query(search_query)
                next_page = self._retrieve_next_page(doc_embeddings, query_embedding, allowed_pages, selected_pages)
            except Exception as exc:
                trace['error'] = str(exc)
                iteration_trace.append(trace)
                return selected_pages, iteration_trace, 'fallback_no_new_page'

            if next_page is None:
                iteration_trace.append(trace)
                return selected_pages, iteration_trace, 'fallback_no_new_page'

            trace['added_page'] = next_page
            trace['retrieval_score'] = next_page['score']
            selected_pages.append(next_page)
            iteration_trace.append(trace)

        return selected_pages, iteration_trace, 'fallback_max_iterations'

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
            log_prefix='M3DocRAG iterate-query evaluator',
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
            'search_query': str(payload.get('search_query') or ''),
        }

    def _evaluator_content(self, sample, pages, benchmark_name):
        prompt = (
            'You are judging whether the provided document page images contain enough evidence '
            'to answer the question. Use only the images. Return strict JSON only with this schema: '
            '{"answerable": true/false, "reason": "...", "missing_evidence": "...", "search_query": "..."} '
            'If answerable is false, write a concise search_query for retrieving the next page from the same PDF. '
            'If answerable is true, search_query may be empty.\n'
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

    def _retrieve_next_page(self, doc_embeddings, query_embedding, allowed_pages, selected_pages):
        selected_indices = {page['page_index'] for page in selected_pages}
        candidate_indices = [page_index for page_index in allowed_pages if page_index not in selected_indices]
        if not candidate_indices:
            return None

        query_embedding = self._normalize_query_embedding(query_embedding).to(dtype=torch.float32)
        candidate_page_embs = doc_embeddings[candidate_indices].to(dtype=torch.float32)
        sim = torch.einsum('qd,ptd->qpt', query_embedding, candidate_page_embs)
        scores = sim.max(dim=2).values.sum(dim=0)
        top_score, top_offset = torch.topk(scores, k=1, largest=True, sorted=True)
        page_index = int(candidate_indices[int(top_offset[0].item())])
        return {
            'page_index': page_index,
            'page_number': page_index + 1,
            'score': float(top_score[0].item()),
        }

    def _encode_search_query(self, search_query):
        retrieval_model = None
        try:
            retrieval_model = self._load_online_retrieval_model()
            query_embs = retrieval_model.encode_queries(
                [search_query],
                batch_size=self.online_colpali['query_batch_size'],
                to_cpu=True,
            )
            return self._normalize_query_embedding(query_embs)
        finally:
            del retrieval_model
            self._release_online_model_memory()

    def _load_online_retrieval_model(self):
        target_device = self._resolve_device(self.online_colpali['device'])
        m3docrag_src = Path(__file__).resolve().parent / 'm3docrag' / 'src'
        if str(m3docrag_src) not in sys.path:
            sys.path.insert(0, str(m3docrag_src))
        from m3docrag.retrieval import ColPaliRetrievalModel

        logger.debug(
            'Loading online ColPali model for one search query. backbone=%s adapter=%s device=%s',
            self.online_colpali['backbone_path'],
            self.online_colpali['adapter_path'],
            target_device,
        )
        noisy_logger_names = ('transformers', 'huggingface_hub', 'accelerate', 'peft', 'colpali_engine')
        previous_levels = {name: logging.getLogger(name).level for name in noisy_logger_names}
        try:
            for name in noisy_logger_names:
                logging.getLogger(name).setLevel(logging.ERROR)
            with open(os.devnull, 'w', encoding='utf-8') as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    retrieval_model = ColPaliRetrievalModel(
                        backbone_name_or_path=self.online_colpali['backbone_path'],
                        adapter_name_or_path=self.online_colpali['adapter_path'],
                    )
                    retrieval_model.model.to(target_device)
        finally:
            for name, level in previous_levels.items():
                logging.getLogger(name).setLevel(level)
        self._online_device = target_device
        return retrieval_model

    def _release_online_model_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _resolve_device(self, device):
        if device != 'auto':
            return device
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def _normalize_query_embedding(self, query_embs):
        if isinstance(query_embs, (list, tuple)):
            query_emb = query_embs[0]
        else:
            query_emb = query_embs
            if getattr(query_emb, 'ndim', None) == 3:
                query_emb = query_emb[0]
        if not isinstance(query_emb, torch.Tensor):
            query_emb = torch.as_tensor(query_emb)
        if query_emb.ndim != 2:
            raise ValueError(f'Expected search query embedding shape [q_tokens, dim], got {tuple(query_emb.shape)}')
        return query_emb.to(dtype=torch.float32, device='cpu')

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
        metadata = super()._metadata(
            candidate_pages,
            retrieved_pages,
            iteration_trace,
            stopped_by,
            allowed_pages,
            pdf_embedding_path,
            question_embedding_path,
        )
        metadata['online_colpali'] = {
            'backbone_path': self.online_colpali['backbone_path'],
            'adapter_path': self.online_colpali['adapter_path'],
            'device': self.online_colpali['device'],
            'resolved_device': self._online_device,
            'query_batch_size': self.online_colpali['query_batch_size'],
        }
        return metadata
