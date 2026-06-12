from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from analysis.results_loader import RunRecord
from analysis.results_metrics import official_score


PRIMARY_MAGERAG_CONTROLLER_MODES = {"full", "full_magerag"}
PRIMARY_MAGERAG_GRAPH_MODE = "full_graph"
OFFICIAL_BENCHMARKS = {"longdocurl", "mmlongbench"}


@dataclass(frozen=True)
class RunAudit:
    run_id: str
    benchmark: str
    baseline: str
    stem: str
    status: str
    paper_excluded: bool
    warnings: tuple[str, ...]
    jsonl_path: str | None
    metrics_path: str | None
    jsonl_size_bytes: int | None
    metrics_size_bytes: int | None
    sample_count: int | None
    completed_count: int | None
    failed_count: int | None
    metric_keys: tuple[str, ...]
    parameters: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "benchmark": self.benchmark,
            "baseline": self.baseline,
            "stem": self.stem,
            "status": self.status,
            "paper_excluded": self.paper_excluded,
            "warnings": list(self.warnings),
            "jsonl_path": self.jsonl_path,
            "metrics_path": self.metrics_path,
            "jsonl_size_bytes": self.jsonl_size_bytes,
            "metrics_size_bytes": self.metrics_size_bytes,
            "sample_count": self.sample_count,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "metric_keys": list(self.metric_keys),
            "parameters": self.parameters,
        }


def normalize_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            value = ast.literal_eval(text)
        except Exception:
            value = [text]
    if isinstance(value, dict):
        value = list(value.values())
    if not isinstance(value, (list, tuple, set)):
        value = [value]

    result: list[int] = []
    for item in value:
        if isinstance(item, dict):
            for key in ("page_number", "page_index", "page", "page_id"):
                if key in item:
                    item = item[key]
                    break
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result


def audit_run(run: RunRecord) -> RunAudit:
    metrics = run.metrics or {}
    warnings: list[str] = []
    paper_excluded = _is_paper_excluded(run)

    if not run.metrics_path:
        status = "missing_metrics"
        warnings.append("missing metrics file")
    elif run.sample_count is not None and run.completed_count == 0:
        status = "empty"
        warnings.append("zero completed samples")
    elif run.sample_count is not None and run.completed_count is not None and run.completed_count < run.sample_count:
        status = "partial"
        warnings.append(f"partial completion: {run.completed_count}/{run.sample_count}")
    elif official_score(metrics, "acc") is None:
        status = "invalid_schema"
        warnings.append("missing official accuracy metric")
    else:
        status = "complete"

    if run.benchmark not in OFFICIAL_BENCHMARKS:
        paper_excluded = True
        warnings.append("benchmark excluded from paper artifacts")
    if run.baseline == "backup" or "/backup/" in _path_text(run.metrics_path, run.jsonl_path):
        paper_excluded = True
        warnings.append("backup result excluded")
    if run.baseline == "magerag" and not _is_primary_magerag_run(run):
        warnings.append("non-primary MAGE-RAG variant")

    return RunAudit(
        run_id=run.run_id,
        benchmark=run.benchmark,
        baseline=run.baseline,
        stem=run.stem,
        status=status,
        paper_excluded=paper_excluded,
        warnings=tuple(dict.fromkeys(warnings)),
        jsonl_path=str(run.jsonl_path) if run.jsonl_path else None,
        metrics_path=str(run.metrics_path) if run.metrics_path else None,
        jsonl_size_bytes=run.jsonl_size_bytes,
        metrics_size_bytes=run.metrics_size_bytes,
        sample_count=run.sample_count,
        completed_count=run.completed_count,
        failed_count=run.failed_count,
        metric_keys=tuple(sorted(metrics.keys())),
        parameters=dict(run.parameters or {}),
    )


def paper_run_diagnostics(runs: Iterable[RunRecord]) -> list[dict[str, Any]]:
    return [audit_run(run).as_dict() for run in sorted(runs, key=lambda item: item.run_id)]


def official_paper_runs(runs: Iterable[RunRecord]) -> dict[tuple[str, str], RunRecord]:
    selected: dict[tuple[str, str], RunRecord] = {}
    for run in runs:
        audit = audit_run(run)
        if audit.paper_excluded or audit.status not in {"complete", "partial"}:
            continue
        if run.baseline == "magerag" and not _is_primary_magerag_run(run):
            continue
        score = official_score(run.metrics or {}, "acc")
        if score is None:
            continue
        key = (run.benchmark, run.baseline)
        if key not in selected or _paper_sort_key(run) > _paper_sort_key(selected[key]):
            selected[key] = run
    return selected


def selected_run_manifest(runs: Iterable[RunRecord]) -> list[dict[str, Any]]:
    selected = official_paper_runs(runs)
    rows = []
    for key, run in sorted(selected.items()):
        audit = audit_run(run)
        rows.append(
            {
                "benchmark": key[0],
                "baseline": key[1],
                "run_id": run.run_id,
                "status": audit.status,
                "score": official_score(run.metrics or {}, "acc"),
                "jsonl_path": str(run.jsonl_path) if run.jsonl_path else None,
                "metrics_path": str(run.metrics_path) if run.metrics_path else None,
                "parameters": dict(run.parameters or {}),
            }
        )
    return rows


def _paper_sort_key(run: RunRecord) -> tuple[int, float, int, str]:
    completed = int(run.completed_count or 0)
    score = official_score(run.metrics or {}, "acc")
    is_complete = int(run.sample_count is not None and run.completed_count == run.sample_count)
    return (is_complete, float(score or -1.0), completed, run.run_id)


def _is_paper_excluded(run: RunRecord) -> bool:
    return run.benchmark not in OFFICIAL_BENCHMARKS or run.baseline == "backup"


def _is_primary_magerag_run(run: RunRecord) -> bool:
    params = run.parameters or {}
    controller_mode = str(params.get("controller_mode") or "")
    graph_mode = str(params.get("graph_mode") or "")
    return controller_mode in PRIMARY_MAGERAG_CONTROLLER_MODES and graph_mode == PRIMARY_MAGERAG_GRAPH_MODE


def _path_text(*paths: Path | None) -> str:
    return " ".join(str(path) for path in paths if path is not None)
