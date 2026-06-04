from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    display_name: str | None = None
    value_type: str = "str"
    numeric: bool = False
    chart_role: str | None = None


@dataclass(frozen=True)
class ChartSpec:
    kind: str
    title: str
    x: str | None = None
    y: str = "score"
    color: str | None = "baseline"
    row: str | None = None
    column: str | None = None
    filters: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "title": self.title,
            "x": self.x,
            "y": self.y,
            "color": self.color,
            "row": self.row,
            "column": self.column,
            "filters": list(self.filters),
        }


@dataclass(frozen=True)
class RunContext:
    benchmark: str
    baseline: str
    stem: str
    jsonl_path: Path | None
    metrics_path: Path | None
    metrics: dict[str, Any]
    first_record: dict[str, Any] | None = None


class AnalysisPlugin:
    name = "base"
    version = "1"
    baseline_names: tuple[str, ...] = ()

    def matches(self, baseline: str) -> bool:
        return baseline in self.baseline_names

    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return ()

    def extract_parameters(self, context: RunContext) -> tuple[dict[str, Any], dict[str, str]]:
        params: dict[str, Any] = {}
        sources: dict[str, str] = {}
        self._merge_params(params, sources, _run_metadata_params(context.metrics), "run_metadata.baselines.params")
        self._merge_params(params, sources, _legacy_metric_params(context.metrics), "metrics.parameters")
        metadata = (context.first_record or {}).get("prepare_metadata") or {}
        if isinstance(metadata, dict):
            self._merge_params(params, sources, self._metadata_params(metadata), "sample.prepare_metadata")
        return params, sources

    def chart_specs(self) -> tuple[ChartSpec, ...]:
        numeric_specs = [spec for spec in self.parameter_specs() if spec.numeric]
        if not numeric_specs:
            return ()
        return tuple(
            ChartSpec(kind="scatter", title=f"{spec.display_name or spec.name} vs score", x=spec.name)
            for spec in numeric_specs
        )

    def diagnostic_rows(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return []

    def case_columns(self) -> tuple[str, ...]:
        return ()

    def has_case_visualization(self) -> bool:
        return False

    def case_visualization(self, record: dict[str, Any]) -> dict[str, Any]:
        return {}

    def _metadata_params(self, metadata: dict[str, Any]) -> dict[str, Any]:
        specs = self.parameter_specs()
        if specs:
            keys = {spec.name for spec in specs}
            return {key: metadata[key] for key in keys if key in metadata}
        return {
            key: value
            for key, value in metadata.items()
            if _looks_like_parameter(key, value)
        }

    @staticmethod
    def _merge_params(
        params: dict[str, Any],
        sources: dict[str, str],
        incoming: dict[str, Any],
        source: str,
    ) -> None:
        for key, value in incoming.items():
            if key not in params:
                params[key] = value
                sources[key] = source


def _run_metadata_params(metrics: dict[str, Any]) -> dict[str, Any]:
    run_metadata = metrics.get("run_metadata")
    if not isinstance(run_metadata, dict):
        return {}
    baselines = run_metadata.get("baselines")
    if not isinstance(baselines, dict):
        return {}
    params = baselines.get("params")
    return dict(params) if isinstance(params, dict) else {}


def _legacy_metric_params(metrics: dict[str, Any]) -> dict[str, Any]:
    params = metrics.get("parameters")
    return dict(params) if isinstance(params, dict) else {}


def _looks_like_parameter(key: str, value: Any) -> bool:
    if isinstance(value, dict | list):
        return False
    suffixes = ("_k", "_b", "_size", "_overlap", "_weight", "_name", "_source")
    return key in {"top_k", "temperature", "max_iterations", "rerank_top_k", "encoder_name"} or key.endswith(suffixes)
