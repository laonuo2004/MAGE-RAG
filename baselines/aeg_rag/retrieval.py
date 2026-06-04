from __future__ import annotations

from baselines.m3docrag import m3docragContextBuilder
from benchmarks.utils.data_utils import colpali_pdf_embeddings_path, colpali_question_embeddings_path, mmlongbench_file_id
from safetensors.torch import load_file
from utils.config_utils import get_config_value


class ColPaliTop1Retriever:
    _load_embeddings = m3docragContextBuilder._load_embeddings
    _retrieve_pages = m3docragContextBuilder._retrieve_pages

    def __init__(self, cfg):
        self.cfg = cfg
        self.top_k = int(get_config_value(cfg, "baselines.agent.initial_retrieval_top_k", 15))
        self.top_k_longdocurl = int(get_config_value(cfg, "baselines.agent.initial_retrieval_top_k_longdocurl", self.top_k))
        self.top_k_mmlongbench = int(get_config_value(cfg, "baselines.agent.initial_retrieval_top_k_mmlongbench", self.top_k))

    def top_k_for(self, benchmark_name: str) -> int:
        if benchmark_name == "longdocurl":
            return self.top_k_longdocurl
        if benchmark_name == "mmlongbench":
            return self.top_k_mmlongbench
        return self.top_k

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
        previous_top_k = self.top_k
        try:
            self.top_k = self.top_k_for(benchmark_name)
            pages = self._retrieve_pages(doc_embeddings, query_embedding, allowed_pages)
        finally:
            self.top_k = previous_top_k
        metadata = {
            "doc_key": doc_key,
            "retrieved_pages": pages,
            "embedding_paths": {"pdf": str(pdf_embedding_path), "question": str(question_embedding_path)},
        }
        return pages, metadata
