from typing import Any, Dict


def build_logical_cost(**overrides) -> Dict[str, Any]:
    logical_cost = {
        "num_llm_calls": 0,
        "num_retriever_calls": 0,
        "num_embedding_calls": 0,
        "estimated_input_tokens": None,
        "num_input_text_chars": 0,
        "num_input_images": 0,
        "num_context_pages": 0,
        "num_context_nodes": 0,
        "num_retrieved_pages": 0,
        "num_retrieved_chunks": 0,
        "num_final_evidence_units": 0,
    }
    logical_cost.update(overrides)
    return logical_cost


def build_retrieval_metadata(**overrides) -> Dict[str, Any]:
    retrieval = {
        "retrieved_items": [],
        "initial_retrieved_pages": [],
        "final_context_pages": [],
        "initial_hit_answer_page": None,
        "final_hit_answer_page": None,
    }
    retrieval.update(overrides)
    return retrieval


def build_context_summary(**overrides) -> Dict[str, Any]:
    context_summary = {
        "page_ids": [],
        "node_ids": [],
        "node_types": [],
        "num_context_pages": 0,
        "num_context_nodes": 0,
        "num_text_units": 0,
        "num_image_units": 0,
        "num_text_chars": 0,
    }
    context_summary.update(overrides)
    return context_summary


class ContextMessages(list):

    def __init__(self, messages, metadata: Dict[str, Any] | None = None):
        super().__init__(messages)
        self.metadata = metadata or {}


class ContextBuilder:
    name = None

    def __init__(self, cfg=None):
        self.cfg = cfg

    def build(self, benchmark_name, sample, **kwargs):
        if benchmark_name == 'mmlongbench':
            return self.build_mmlongbench(sample, **kwargs)
        if benchmark_name == 'longdocurl':
            return self.build_longdocurl(sample, **kwargs)
        raise ValueError(f'Unsupported benchmark for context builder: {benchmark_name}')

    def build_mmlongbench(self, sample, **kwargs):
        raise NotImplementedError

    def build_longdocurl(self, sample, **kwargs):
        raise NotImplementedError
