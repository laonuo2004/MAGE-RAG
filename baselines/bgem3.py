import json
import os
import torch
from safetensors.torch import load_file

from baselines.base import ContextBuilder, ContextMessages
from benchmarks.utils.document_preprocess import (
    bgem3_doc_cache_variant,
    bgem3_query_cache_variant,
)
from benchmarks.utils.data_utils import bgem3_cache_root, mmlongbench_file_id
from utils.config_utils import get_config_value, require_config_value
from utils.llm_utils import text_content_parts


TEXT_SYSTEM_PROMPT = (
    "You are an expert in document question-answering. "
    "Answer the question using only the retrieved OCR/text chunks from the document. "
    "If the answer cannot be found, say Not answerable.\n"
)


class BGEM3ContextBuilder(ContextBuilder):
    name = "bgem3"

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.mode = str(get_config_value(self.cfg, "baselines.params.mode", "dense"))
        self.text_source = str(get_config_value(self.cfg, "baselines.params.text_source", "ocr"))
        self.top_k = int(get_config_value(self.cfg, "baselines.params.top_k", 5))
        self.chunk_size = int(get_config_value(self.cfg, "baselines.params.chunk_size", 200))
        self.chunk_overlap = int(get_config_value(self.cfg, "baselines.params.chunk_overlap", 20))
        self.allow_cross_page = bool(get_config_value(self.cfg, "baselines.params.allow_cross_page", True))
        self.max_cross_pages = get_config_value(self.cfg, "baselines.params.max_cross_pages", None)
        if self.max_cross_pages is not None:
            self.max_cross_pages = int(self.max_cross_pages)
        self.checkpoint = require_config_value(self.cfg, "baselines.checkpoint")
        self.max_context_chars = get_config_value(self.cfg, "baselines.max_context_chars", None)
        if self.max_context_chars is not None:
            self.max_context_chars = int(self.max_context_chars)

        if self.mode != "dense":
            raise ValueError("BGEM3 first version only supports baselines.params.mode=dense.")
        if self.text_source not in {"ocr", "vlm_text"}:
            raise ValueError("BGEM3 currently only supports baselines.params.text_source in {ocr, vlm_text}.")
        if self.top_k <= 0:
            raise ValueError("BGEM3 baseline requires cfg.baselines.params.top_k > 0.")
        if self.chunk_size <= 0:
            raise ValueError("BGEM3 baseline requires cfg.baselines.params.chunk_size > 0.")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("BGEM3 baseline requires 0 <= cfg.baselines.params.chunk_overlap < chunk_size.")

        self.doc_cache_variant = bgem3_doc_cache_variant(
            self.checkpoint,
            self.mode,
            self.text_source,
            self.chunk_size,
            self.chunk_overlap,
            self.allow_cross_page,
            self.max_cross_pages,
        )
        self.query_cache_variant = bgem3_query_cache_variant(
            self.checkpoint,
            self.mode,
        )

    def build_mmlongbench(self, sample, **kwargs):
        doc_key = self._mmlongbench_doc_key(sample["doc_id"])
        query_key = str(sample["question_id"])
        allowed_pages = None
        retrieval = self._retrieve("mmlongbench", doc_key, query_key, allowed_pages)
        prompt = self._build_prompt(sample["question"], retrieval)
        return ContextMessages(
            [{"role": "user", "content": text_content_parts(prompt)}],
            metadata=self._metadata(retrieval, allowed_pages, doc_key, query_key),
        )

    def build_longdocurl(self, sample, **kwargs):
        images = sample.get("images")
        allowed_pages = None
        if isinstance(images, str):
            images = [images]
        if images:
            allowed_pages = []
            for image_path in images:
                filename = os.path.basename(str(image_path))
                if "_" not in filename:
                    continue
                page_part = filename.rsplit("_", 1)[-1].split(".", 1)[0]
                if page_part.isdigit():
                    allowed_pages.append(int(page_part))
            allowed_pages = sorted(set(allowed_pages)) or None
        if allowed_pages is None:
            allowed_pages = sample.get("start_end_idx")
        doc_key = str(sample["question_id"])
        query_key = str(sample["question_id"])
        retrieval = self._retrieve("longdocurl", doc_key, query_key, allowed_pages)
        prompt = self._build_prompt(sample["question"], retrieval)
        return ContextMessages(
            [{"role": "user", "content": text_content_parts(prompt)}],
            metadata=self._metadata(retrieval, allowed_pages, doc_key, query_key),
        )

    def _retrieve(self, benchmark_name, doc_key, query_key, allowed_pages):
        doc_embedding_path = self._resolve_path(benchmark_name, "doc_embeddings", self.doc_cache_variant, doc_key, ".safetensors")
        query_embedding_path = self._resolve_path(benchmark_name, "query_embeddings", self.query_cache_variant, query_key, ".safetensors")
        chunk_metadata_path = self._resolve_path(benchmark_name, "chunk_metadata", self.doc_cache_variant, doc_key, ".json")

        doc_embeddings, query_embedding = self._load_embeddings(doc_embedding_path, query_embedding_path)
        metadata = self._load_chunk_metadata(chunk_metadata_path)
        candidate_indices = self._candidate_chunk_indices(metadata, allowed_pages)
        if not candidate_indices:
            candidate_indices = list(range(len(metadata)))
        candidate_doc_embs = doc_embeddings[candidate_indices]
        scores = candidate_doc_embs @ query_embedding.to(dtype=candidate_doc_embs.dtype)
        top_count = min(self.top_k, len(candidate_indices))
        top_scores, top_offsets = torch.topk(scores, k=top_count, largest=True, sorted=True)

        retrieval = []
        for rank, (score, offset) in enumerate(zip(top_scores.tolist(), top_offsets.tolist()), start=1):
            record = metadata[candidate_indices[int(offset)]]
            retrieval.append({
                "rank": rank,
                "chunk_id": int(record["chunk_id"]),
                "score": float(score),
                "text": record["text"],
                "chunk_index": int(record["chunk_index"]),
                "start_page_index": int(record["start_page_index"]),
                "end_page_index": int(record["end_page_index"]),
                "start_page_number": int(record["start_page_number"]),
                "end_page_number": int(record["end_page_number"]),
                "covered_page_indices": [int(v) for v in record["covered_page_indices"]],
                "covered_page_numbers": [int(v) for v in record["covered_page_numbers"]],
            })
        return retrieval

    def _load_embeddings(self, doc_embedding_path, query_embedding_path):
        if not os.path.exists(doc_embedding_path):
            raise FileNotFoundError(f"Missing BGEM3 document embeddings: {doc_embedding_path}")
        if not os.path.exists(query_embedding_path):
            raise FileNotFoundError(f"Missing BGEM3 query embeddings: {query_embedding_path}")

        doc_tensors = load_file(doc_embedding_path, device="cpu")
        query_tensors = load_file(query_embedding_path, device="cpu")
        if "dense_vecs" not in doc_tensors:
            raise KeyError(f'Document embedding file missing "dense_vecs": {doc_embedding_path}')
        if "query_dense_vec" not in query_tensors:
            raise KeyError(f'Query embedding file missing "query_dense_vec": {query_embedding_path}')

        doc_embeddings = doc_tensors["dense_vecs"].to(dtype=torch.float32)
        query_embedding = query_tensors["query_dense_vec"].to(dtype=torch.float32)
        if doc_embeddings.ndim != 2:
            raise ValueError(
                f"Expected BGEM3 document embeddings shape [n_chunks, dim], got {tuple(doc_embeddings.shape)}"
            )
        if query_embedding.ndim == 2 and query_embedding.shape[0] == 1:
            query_embedding = query_embedding[0]
        if query_embedding.ndim != 1:
            raise ValueError(
                f"Expected BGEM3 query embedding shape [dim], got {tuple(query_embedding.shape)}"
            )
        if doc_embeddings.shape[-1] != query_embedding.shape[-1]:
            raise ValueError(
                "Document and query embedding dims differ: "
                f"{doc_embeddings.shape[-1]} != {query_embedding.shape[-1]}"
            )
        return doc_embeddings, query_embedding

    def _load_chunk_metadata(self, chunk_metadata_path):
        if not os.path.exists(chunk_metadata_path):
            raise FileNotFoundError(f"Missing BGEM3 chunk metadata: {chunk_metadata_path}")
        with open(chunk_metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _candidate_chunk_indices(self, metadata, allowed_pages):
        if allowed_pages is None:
            return list(range(len(metadata)))
        if isinstance(allowed_pages, (list, tuple)) and len(allowed_pages) == 2 and all(
            isinstance(value, int) for value in allowed_pages
        ):
            allowed_page_set = set(range(int(allowed_pages[0]), int(allowed_pages[1]) + 1))
        else:
            allowed_page_set = set(int(page) for page in allowed_pages)
        indices = []
        for idx, record in enumerate(metadata):
            covered_pages = {int(page) for page in record["covered_page_indices"]}
            if covered_pages & allowed_page_set:
                indices.append(idx)
        return indices

    def _build_prompt(self, question, retrieval):
        chunk_blocks = []
        total_chars = 0
        for item in retrieval:
            if item["start_page_number"] == item["end_page_number"]:
                page_label = f'Page {item["start_page_number"]}'
            else:
                page_label = f'Pages {item["start_page_number"]}-{item["end_page_number"]}'
            block = (
                f'[{page_label} | Chunk {item["chunk_index"] + 1} | BGE-M3 dense score {item["score"]:.4f}]\n'
                f'{item["text"]}'
            )
            if self.max_context_chars is not None and total_chars + len(block) > self.max_context_chars:
                break
            chunk_blocks.append(block)
            total_chars += len(block)
        retrieved_text = "\n\n".join(chunk_blocks)
        return (
            TEXT_SYSTEM_PROMPT
            + "\nQuestion:\n"
            + str(question)
            + "\n\nRetrieved OCR/text chunks:\n"
            + retrieved_text
        )

    def _metadata(self, retrieval, allowed_pages, doc_key, query_key):
        retrieved_pages = []
        best_by_page = {}
        for item in retrieval:
            for page_index, page_number in zip(item["covered_page_indices"], item["covered_page_numbers"]):
                current = best_by_page.get(page_index)
                if current is None or item["score"] > current["score"]:
                    best_by_page[page_index] = {
                        "page_index": page_index,
                        "page_number": page_number,
                        "score": item["score"],
                    }
        for page_index in sorted(best_by_page):
            retrieved_pages.append(best_by_page[page_index])
        return {
            "context_builder": self.name,
            "retrieved_chunks": retrieval,
            "retrieved_pages": retrieved_pages,
            "allowed_pages": [] if allowed_pages is None else list(allowed_pages),
            "mode": self.mode,
            "text_source": self.text_source,
            "top_k": self.top_k,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "allow_cross_page": self.allow_cross_page,
            "max_cross_pages": self.max_cross_pages,
            "doc_cache_variant": self.doc_cache_variant,
            "query_cache_variant": self.query_cache_variant,
            "doc_key": doc_key,
            "query_key": query_key,
        }

    def _resolve_path(self, benchmark_name, cache_kind, variant, stem, suffix):
        return os.path.join(str(bgem3_cache_root(benchmark_name, cache_kind)), variant, f"{stem}{suffix}")

    def _mmlongbench_doc_key(self, doc_id):
        return mmlongbench_file_id(doc_id)
