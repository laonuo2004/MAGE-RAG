import os

import torch
from safetensors.torch import load_file

from .base import ContextBuilder, ContextMessages
from .image import VISION_SYSTEM_PROMPT
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


class m3docragContextBuilder(ContextBuilder):
    name = 'm3docrag'

    def __init__(self, cfg=None):
        super().__init__(cfg)
        top_k = require_config_value(self.cfg, 'baselines.params.top_k')
        self.top_k = int(top_k)
        if self.top_k <= 0:
            raise ValueError('M3DocRAG baseline requires cfg.baselines.params.top_k > 0.')

    def build_mmlongbench(self, sample, **kwargs):
        doc_id = sample['doc_id']
        question_id = sample['question_id']
        pdf_embedding_path = colpali_pdf_embeddings_path('mmlongbench', mmlongbench_file_id(doc_id))
        question_embedding_path = colpali_question_embeddings_path('mmlongbench', question_id)
        doc_embeddings, query_embedding = self._load_embeddings(pdf_embedding_path, question_embedding_path)
        benchmark_cfg = require_config_value(self.cfg, 'benchmarks')
        allowed_pages = allowed_page_indices('mmlongbench', sample, benchmark_cfg, doc_embeddings.shape[0])
        retrieval = self._retrieve_pages(doc_embeddings, query_embedding, allowed_pages)

        from PIL import Image

        content = [{'type': 'text', 'text': sample['question']}]
        for page in retrieval:
            page_path = mmlongbench_png_page_path(benchmark_cfg, doc_id, page['page_index'])
            if not os.path.exists(page_path):
                raise FileNotFoundError(f'Missing preprocessed PNG page for doc_id={doc_id}: {page_path}')
            with Image.open(page_path) as image:
                encoded_image = encode_pil_image_to_base64(image)
            content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{encoded_image}'}})

        return ContextMessages(
            [{'role': 'user', 'content': content}],
            metadata=self._metadata(retrieval, allowed_pages, pdf_embedding_path, question_embedding_path),
        )

    def build_longdocurl(self, sample, **kwargs):
        doc_no = sample['doc_no']
        question_id = sample['question_id']
        pdf_embedding_path = colpali_pdf_embeddings_path('longdocurl', doc_no)
        question_embedding_path = colpali_question_embeddings_path('longdocurl', question_id)
        doc_embeddings, query_embedding = self._load_embeddings(pdf_embedding_path, question_embedding_path)
        benchmark_cfg = require_config_value(self.cfg, 'benchmarks')
        allowed_pages = allowed_page_indices('longdocurl', sample, benchmark_cfg, doc_embeddings.shape[0])
        retrieval = self._retrieve_pages(doc_embeddings, query_embedding, allowed_pages)

        prompt = (
            VISION_SYSTEM_PROMPT
            + 'Following is our question: \n'
            + f'<question>{sample["question"]}</question>'
            + '\n'
        )
        content = [{'type': 'text', 'text': prompt}]
        for index, page in enumerate(retrieval):
            image_path = self._longdocurl_image_path(doc_no, page['page_index'])
            content.extend([
                {'type': 'text', 'text': f'Below is the {index + 1}-th image (total {len(retrieval)} images).\n'},
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{encode_image_file_to_base64(image_path)}'}},
            ])

        return ContextMessages(
            [{'role': 'user', 'content': content}],
            metadata=self._metadata(retrieval, allowed_pages, pdf_embedding_path, question_embedding_path),
        )

    def _load_embeddings(self, pdf_embedding_path, question_embedding_path):
        if not os.path.exists(pdf_embedding_path):
            raise FileNotFoundError(f'Missing PDF ColPali embedding: {pdf_embedding_path}')
        if not os.path.exists(question_embedding_path):
            raise FileNotFoundError(f'Missing question ColPali embedding: {question_embedding_path}')

        pdf_tensors = load_file(pdf_embedding_path, device='cpu')
        question_tensors = load_file(question_embedding_path, device='cpu')
        if 'embeddings' not in pdf_tensors:
            raise KeyError(f'PDF embedding file missing "embeddings": {pdf_embedding_path}')
        if 'query_embedding' not in question_tensors:
            raise KeyError(f'Question embedding file missing "query_embedding": {question_embedding_path}')

        doc_embeddings = pdf_tensors['embeddings'].to(dtype=torch.float32)
        query_embedding = question_tensors['query_embedding'].to(dtype=torch.float32)
        if doc_embeddings.ndim != 3:
            raise ValueError(f'Expected PDF embeddings shape [n_pages, n_tokens, dim], got {tuple(doc_embeddings.shape)}')
        if query_embedding.ndim == 3 and query_embedding.shape[0] == 1:
            query_embedding = query_embedding[0]
        if query_embedding.ndim != 2:
            raise ValueError(f'Expected query_embedding shape [q_tokens, dim], got {tuple(query_embedding.shape)}')
        if doc_embeddings.shape[-1] != query_embedding.shape[-1]:
            raise ValueError(
                'PDF and query embedding dims differ: '
                f'{doc_embeddings.shape[-1]} != {query_embedding.shape[-1]}'
            )
        return doc_embeddings, query_embedding

    def _retrieve_pages(self, doc_embeddings, query_embedding, allowed_pages):
        if not allowed_pages:
            raise ValueError('M3DocRAG retrieval has no allowed pages after applying benchmark page mask.')
        candidate_page_embs = doc_embeddings[allowed_pages]
        sim = torch.einsum('qd,ptd->qpt', query_embedding, candidate_page_embs)
        scores = sim.max(dim=2).values.sum(dim=0)
        top_count = min(self.top_k, len(allowed_pages))
        top_scores, top_offsets = torch.topk(scores, k=top_count, largest=True, sorted=True)
        retrieved_pages = []
        for score, offset in zip(top_scores.tolist(), top_offsets.tolist()):
            page_index = int(allowed_pages[int(offset)])
            retrieved_pages.append({
                'page_index': page_index,
                'page_number': page_index + 1,
                'score': float(score),
            })
        return retrieved_pages

    def _longdocurl_image_path(self, doc_no, page_index):
        return os.path.join(
            str(require_config_value(self.cfg, 'benchmarks.image_prefix')),
            doc_no[:4],
            f'{doc_no}_{page_index}.png',
        )

    def _metadata(self, retrieved_pages, allowed_pages, pdf_embedding_path, question_embedding_path):
        return {
            'context_builder': self.name,
            'retrieved_pages': retrieved_pages,
            'allowed_pages': list(allowed_pages),
            'top_k': self.top_k,
            'embedding_paths': {
                'pdf': str(pdf_embedding_path),
                'question': str(question_embedding_path),
            },
        }
