import os

import torch
from safetensors.torch import load_file

from .base import ContextBuilder, ContextMessages
from benchmarks.utils.document_preprocess import (
    allowed_page_indices,
    encode_image_file_to_base64,
    encode_pil_image_to_base64,
)
from benchmarks.utils.data_utils import (
    mmlongbench_file_id,
    mmlongbench_png_page_path,
    visrag_pdf_embeddings_path,
    visrag_question_embeddings_path,
)
from utils.config_utils import get_config_value, require_config_value


EVISRAG_EVIDENCE_PROMPT_TEMPLATE = """You are an AI Visual QA assistant. I will provide you with a question and several images. Please follow the four steps below:

Step 1: Observe the Images
First, analyze the question and consider what types of images may contain relevant information. Then, examine each image one by one, paying special attention to aspects related to the question. Identify whether each image contains any potentially relevant information.
Wrap your observations within <observe></observe> tags.

Step 2: Record Evidences from Images
After reviewing all images, record the evidence you find for each image within <evidence></evidence> tags.
If you are certain that an image contains no relevant information, record it as: [i]: no relevant information(where i denotes the index of the image).
If an image contains relevant evidence, record it as: [j]: [the evidence you find for the question](where j is the index of the image).

Step 3: Reason Based on the Question and Evidences
Based on the recorded evidences, reason about the answer to the question.
Include your step-by-step reasoning within <think></think> tags.

Step 4: Answer the Question
Provide your final answer based only on the evidences you found in the images.
Wrap your answer within <answer></answer> tags.
Avoid adding unnecessary contents in your final answer, like if the question is a yes/no question, simply answer "yes" or "no".
If none of the images contain sufficient information to answer the question, respond with <answer>insufficient to answer</answer>.

Formatting Requirements:
Use the exact tags <observe>, <evidence>, <think>, and <answer> for structured output.
It is possible that none, one, or several images contain relevant evidence.
If you find no evidence or few evidences, and insufficient to help you answer the question, follow the instruction above for insufficient information.

Question and images are provided below. Please follow the steps as instructed.
Question: {question}
"""


class EVisRAGContextBuilder(ContextBuilder):
    name = 'evisrag'

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.top_k = int(get_config_value(self.cfg, 'baselines.params.top_k', 3))
        self.max_images = int(get_config_value(self.cfg, 'baselines.params.max_images', self.top_k))
        self.prompt_mode = str(get_config_value(self.cfg, 'baselines.params.prompt_mode', 'evidence_grpo'))
        self.longdocurl_shard = str(get_config_value(self.cfg, 'baselines.params.longdocurl_shard', '4000-4999'))
        if self.top_k <= 0:
            raise ValueError('EVisRAG baseline requires cfg.baselines.params.top_k > 0.')
        if self.max_images <= 0:
            raise ValueError('EVisRAG baseline requires cfg.baselines.params.max_images > 0.')
        if self.prompt_mode != 'evidence_grpo':
            raise ValueError('EVisRAG baseline currently supports only baselines.params.prompt_mode=evidence_grpo.')

    def build_mmlongbench(self, sample, **kwargs):
        doc_id = sample['doc_id']
        question_id = sample['question_id']
        doc_key = mmlongbench_file_id(doc_id)
        pdf_embedding_path = visrag_pdf_embeddings_path('mmlongbench', doc_key)
        question_embedding_path = visrag_question_embeddings_path('mmlongbench', question_id)
        doc_embeddings, query_embedding = self._load_embeddings(pdf_embedding_path, question_embedding_path)
        benchmark_cfg = require_config_value(self.cfg, 'benchmarks')
        allowed_pages = allowed_page_indices('mmlongbench', sample, benchmark_cfg, doc_embeddings.shape[0])
        retrieval = self._retrieve_pages(doc_embeddings, query_embedding, allowed_pages)

        from PIL import Image

        content = [{'type': 'text', 'text': self._prompt(sample['question'])}]
        for page in retrieval[:self.max_images]:
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
        pdf_embedding_path = visrag_pdf_embeddings_path('longdocurl', doc_no, shard=self.longdocurl_shard)
        question_embedding_path = visrag_question_embeddings_path('longdocurl', question_id)
        doc_embeddings, query_embedding = self._load_embeddings(pdf_embedding_path, question_embedding_path)
        benchmark_cfg = require_config_value(self.cfg, 'benchmarks')
        allowed_pages = allowed_page_indices('longdocurl', sample, benchmark_cfg, doc_embeddings.shape[0])
        retrieval = self._retrieve_pages(doc_embeddings, query_embedding, allowed_pages)

        content = [{'type': 'text', 'text': self._prompt(sample['question'])}]
        for index, page in enumerate(retrieval[:self.max_images]):
            image_path = self._longdocurl_image_path(doc_no, page['page_index'])
            content.extend([
                {'type': 'text', 'text': f'Below is the {index + 1}-th image (total {min(len(retrieval), self.max_images)} images).\n'},
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{encode_image_file_to_base64(image_path)}'}},
            ])

        return ContextMessages(
            [{'role': 'user', 'content': content}],
            metadata=self._metadata(retrieval, allowed_pages, pdf_embedding_path, question_embedding_path),
        )

    def _load_embeddings(self, pdf_embedding_path, question_embedding_path):
        if not os.path.exists(pdf_embedding_path):
            raise FileNotFoundError(f'Missing VisRAG-Ret PDF embedding: {pdf_embedding_path}')
        if not os.path.exists(question_embedding_path):
            raise FileNotFoundError(f'Missing VisRAG-Ret question embedding: {question_embedding_path}')

        pdf_tensors = load_file(pdf_embedding_path, device='cpu')
        question_tensors = load_file(question_embedding_path, device='cpu')
        if 'embeddings' not in pdf_tensors:
            raise KeyError(f'PDF embedding file missing "embeddings": {pdf_embedding_path}')
        if 'query_embedding' not in question_tensors:
            raise KeyError(f'Question embedding file missing "query_embedding": {question_embedding_path}')

        doc_embeddings = pdf_tensors['embeddings'].to(dtype=torch.float32)
        query_embedding = question_tensors['query_embedding'].to(dtype=torch.float32)
        if doc_embeddings.ndim != 2:
            raise ValueError(f'Expected VisRAG-Ret PDF embeddings shape [n_pages, dim], got {tuple(doc_embeddings.shape)}')
        if query_embedding.ndim == 2 and query_embedding.shape[0] == 1:
            query_embedding = query_embedding[0]
        if query_embedding.ndim != 1:
            raise ValueError(f'Expected VisRAG-Ret query_embedding shape [dim], got {tuple(query_embedding.shape)}')
        if doc_embeddings.shape[-1] != query_embedding.shape[-1]:
            raise ValueError(
                'PDF and query embedding dims differ: '
                f'{doc_embeddings.shape[-1]} != {query_embedding.shape[-1]}'
            )
        return doc_embeddings, query_embedding

    def _retrieve_pages(self, doc_embeddings, query_embedding, allowed_pages):
        if not allowed_pages:
            raise ValueError('EVisRAG retrieval has no allowed pages.')
        candidate_page_embs = doc_embeddings[allowed_pages]
        scores = candidate_page_embs @ query_embedding.to(dtype=candidate_page_embs.dtype)
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

    def _prompt(self, question):
        return EVISRAG_EVIDENCE_PROMPT_TEMPLATE.format(question=question)

    def _longdocurl_image_path(self, doc_no, page_index):
        return os.path.join(
            str(require_config_value(self.cfg, 'benchmarks.image_prefix')),
            doc_no[:4],
            f'{doc_no}_{page_index}.png',
        )

    def _metadata(self, retrieved_pages, allowed_pages, pdf_embedding_path, question_embedding_path):
        return {
            'context_builder': self.name,
            'retrieval_model': 'VisRAG-Ret',
            'generation_prompt': self.prompt_mode,
            'retrieved_pages': retrieved_pages,
            'allowed_pages': list(allowed_pages),
            'top_k': self.top_k,
            'max_images': self.max_images,
            'embedding_paths': {
                'pdf': str(pdf_embedding_path),
                'question': str(question_embedding_path),
            },
        }
