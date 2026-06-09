from __future__ import annotations

import requests
import torch

from baselines.m3docrag import m3docragContextBuilder
from benchmarks.utils.data_utils import colpali_pdf_embeddings_path, colpali_question_embeddings_path, mmlongbench_file_id
from safetensors.torch import load_file
from utils.config_utils import get_config_value


class ColPaliTop1Retriever:
    """
    Stage I initial grounding 的 ColPali 页面检索器。

    这里复用 M3DocRAG 的 embedding 加载和相似度检索实现，让 MAGE-RAG 的差异集中在
    后续 evidence graph expansion，而不是重复实现 page-level retrieval。
    """

    _load_embeddings = m3docragContextBuilder._load_embeddings
    _retrieve_pages = m3docragContextBuilder._retrieve_pages

    def __init__(self, cfg):
        self.cfg = cfg
        self.top_k = int(get_config_value(cfg, "baselines.params.top_k", 5))
        self.online_model = str(get_config_value(cfg, "baselines.online_colpali.model", "colpali-v1.3"))
        self.online_url = str(get_config_value(cfg, "baselines.online_colpali.vllm_url", "http://localhost:8020"))

    def embedding_page_count(self, benchmark_name: str, sample: dict) -> int:
        if benchmark_name == "mmlongbench":
            doc_key = mmlongbench_file_id(sample["doc_id"])
            pdf_embedding_path = colpali_pdf_embeddings_path("mmlongbench", doc_key)
        else:
            doc_key = sample["doc_no"]
            pdf_embedding_path = colpali_pdf_embeddings_path("longdocurl", doc_key)
        pdf_tensors = load_file(pdf_embedding_path, device="cpu")
        if "embeddings" not in pdf_tensors:
            raise KeyError(f'PDF embedding file missing "embeddings": {pdf_embedding_path}')
        return int(pdf_tensors["embeddings"].shape[0])

    def top_k_for(self, benchmark_name: str) -> int:
        return self.top_k

    def retrieve(self, benchmark_name: str, sample: dict, allowed_pages: list[int]) -> tuple[dict, dict]:
        pages, metadata = self.retrieve_many(benchmark_name, sample, allowed_pages)
        return pages[0], metadata

    def retrieve_many(self, benchmark_name: str, sample: dict, allowed_pages: list[int]) -> tuple[list[dict], dict]:
        if benchmark_name == "mmlongbench":
            doc_key = mmlongbench_file_id(sample["doc_id"])
            pdf_embedding_path = colpali_pdf_embeddings_path("mmlongbench", doc_key)
        else:
            doc_key = sample["doc_no"]
            pdf_embedding_path = colpali_pdf_embeddings_path("longdocurl", doc_key)
        question_embedding_path = colpali_question_embeddings_path(benchmark_name, sample["question_id"])
        doc_embeddings, query_embedding = self._load_embeddings(pdf_embedding_path, question_embedding_path)
        pages = self._retrieve_pages(doc_embeddings, query_embedding, allowed_pages)
        metadata = {
            "doc_key": doc_key,
            "retrieved_pages": pages,
            "embedding_paths": {"pdf": str(pdf_embedding_path), "question": str(question_embedding_path)},
        }
        return pages, metadata

    def retrieve_from_query(
        self,
        benchmark_name: str,
        sample: dict,
        query: str,
        allowed_pages: list[int],
        excluded_pages: set[int] | None = None,
    ) -> tuple[list[dict], dict]:
        doc_embeddings, pdf_embedding_path, doc_key = self._load_doc_embeddings(benchmark_name, sample)
        query_embedding = self._encode_online_query(query)
        candidate_pages = [
            int(page_index)
            for page_index in allowed_pages
            if int(page_index) not in set(excluded_pages or set())
        ]
        pages = self._retrieve_pages_from_query_embedding(doc_embeddings, query_embedding, candidate_pages)
        return pages, {
            "doc_key": doc_key,
            "retrieved_pages": pages,
            "query": str(query),
            "embedding_paths": {"pdf": str(pdf_embedding_path)},
            "online_colpali": {"model": self.online_model, "vllm_url": self.online_url},
        }

    def _load_doc_embeddings(self, benchmark_name: str, sample: dict):
        if benchmark_name == "mmlongbench":
            doc_key = mmlongbench_file_id(sample["doc_id"])
            pdf_embedding_path = colpali_pdf_embeddings_path("mmlongbench", doc_key)
        else:
            doc_key = sample["doc_no"]
            pdf_embedding_path = colpali_pdf_embeddings_path("longdocurl", doc_key)
        pdf_tensors = load_file(pdf_embedding_path, device="cpu")
        if "embeddings" not in pdf_tensors:
            raise KeyError(f'PDF embedding file missing "embeddings": {pdf_embedding_path}')
        doc_embeddings = pdf_tensors["embeddings"].to(dtype=torch.float32)
        if doc_embeddings.ndim != 3:
            raise ValueError(f"Expected PDF embeddings shape [n_pages, n_tokens, dim], got {tuple(doc_embeddings.shape)}")
        return doc_embeddings, pdf_embedding_path, doc_key

    def _encode_online_query(self, query: str) -> torch.Tensor:
        response = requests.post(
            f"{self.online_url.rstrip('/')}/pooling",
            json={"model": self.online_model, "task": "token_embed", "input": [str(query or " ")]},
            timeout=180,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(f"ColPali /pooling query request failed: {response.text[:1000]}") from exc
        tensor = torch.tensor(response.json()["data"][0]["data"], dtype=torch.float32)
        if tensor.ndim != 2:
            raise ValueError(f"Expected query embedding [q_tokens, dim], got {tuple(tensor.shape)}")
        return tensor

    def _retrieve_pages_from_query_embedding(self, doc_embeddings, query_embedding, allowed_pages: list[int]) -> list[dict]:
        if not allowed_pages:
            return []
        candidate_page_embs = doc_embeddings[allowed_pages].to(dtype=torch.float32)
        query_embedding = query_embedding.to(dtype=torch.float32)
        if candidate_page_embs.shape[-1] != query_embedding.shape[-1]:
            raise ValueError(
                "PDF and query embedding dims differ: "
                f"{candidate_page_embs.shape[-1]} != {query_embedding.shape[-1]}"
            )
        sim = torch.einsum("qd,ptd->qpt", query_embedding, candidate_page_embs)
        scores = sim.max(dim=2).values.sum(dim=0)
        top_count = min(self.top_k, len(allowed_pages))
        top_scores, top_offsets = torch.topk(scores, k=top_count, largest=True, sorted=True)
        return [
            {
                "page_index": int(allowed_pages[int(offset)]),
                "page_number": int(allowed_pages[int(offset)]) + 1,
                "score": float(score),
            }
            for score, offset in zip(top_scores.tolist(), top_offsets.tolist())
        ]
