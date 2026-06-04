from __future__ import annotations

from typing import Any

from analysis.plugins.base import AnalysisPlugin, ChartSpec, ParameterSpec


class DefaultPlugin(AnalysisPlugin):
    name = "default"
    baseline_names = ("*",)


class BM25Plugin(AnalysisPlugin):
    name = "bm25"
    baseline_names = ("bm25",)

    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec("top_k", "top_k", "int", True, "filter"),
            ParameterSpec("chunk_size", "chunk_size", "int", True, "row"),
            ParameterSpec("chunk_overlap", "chunk_overlap", "int", True, "column"),
            ParameterSpec("bm25_k1", "BM25 k1", "float", True),
            ParameterSpec("bm25_b", "BM25 b", "float", True),
        )

    def chart_specs(self) -> tuple[ChartSpec, ...]:
        return (
            ChartSpec(
                kind="heatmap",
                title="BM25 chunk sweep",
                row="chunk_size",
                column="chunk_overlap",
                filters=("top_k",),
            ),
        )


class DenseRetrieverPlugin(AnalysisPlugin):
    name = "dense_retriever"
    baseline_names = ("bgem3", "colbertv2")

    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec("top_k", "top_k", "int", True),
            ParameterSpec("chunk_size", "chunk_size", "int", True),
            ParameterSpec("chunk_overlap", "chunk_overlap", "int", True),
            ParameterSpec("text_source", "text source", "str"),
            ParameterSpec("allow_cross_page", "allow cross page", "bool"),
            ParameterSpec("max_cross_pages", "max cross pages", "int", True),
        )


class M3DocRAGPlugin(AnalysisPlugin):
    name = "m3docrag"
    baseline_names = ("m3docrag", "m3docrag-colpali-v1_3")

    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return (ParameterSpec("top_k", "top_k", "int", True),)

    def chart_specs(self) -> tuple[ChartSpec, ...]:
        return (ChartSpec(kind="line", title="M3DocRAG top_k", x="top_k", color="baseline"),)


class M3DocRAGIteratePlugin(AnalysisPlugin):
    name = "m3docrag_iterate"
    baseline_names = ("m3docrag-iterate", "m3docrag_iterate", "m3docrag-iterate-query")

    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec("max_iterations", "max iterations", "int", True),
            ParameterSpec("evaluator_model_name", "evaluator model", "str"),
        )

    def chart_specs(self) -> tuple[ChartSpec, ...]:
        return (ChartSpec(kind="line", title="M3DocRAG iterate max_iterations", x="max_iterations", color="baseline"),)

    def diagnostic_rows(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for record in records:
            trace = (record.get("prepare_metadata") or {}).get("iteration_trace")
            if not isinstance(trace, list):
                continue
            for fallback_index, step in enumerate(trace, start=1):
                if not isinstance(step, dict):
                    continue
                rows.append(
                    {
                        "question_id": record.get("question_id"),
                        "iteration": step.get("iteration", fallback_index),
                        "query": step.get("query") or step.get("rewritten_query"),
                        "decision": step.get("decision"),
                        "score": step.get("score"),
                        "retrieved_count": _count_retrieved(step),
                    }
                )
        return rows


class PassthroughPlugin(AnalysisPlugin):
    name = "passthrough"
    baseline_names = ("image", "ocr")


def get_plugins() -> tuple[AnalysisPlugin, ...]:
    return (
        BM25Plugin(),
        DenseRetrieverPlugin(),
        M3DocRAGPlugin(),
        M3DocRAGIteratePlugin(),
        PassthroughPlugin(),
    )


def _count_retrieved(step: dict[str, Any]) -> int:
    count = 0
    for key in ("retrieved_chunks", "retrieved_pages", "retrieved_passages"):
        value = step.get(key)
        if isinstance(value, list):
            count += len(value)
    return count
