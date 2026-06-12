from __future__ import annotations

import ast
from typing import Any

from analysis.plugins import get_plugin
from analysis.result_fields import display_score


def flatten_metrics(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    _flatten_node(metrics, [], rows, None)
    return rows


def build_leaderboard(runs: list[Any], metric: str = "acc") -> list[dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for run in runs:
        score = official_score(getattr(run, "metrics", {}) or {}, metric)
        if score is None:
            continue
        key = (run.benchmark, run.baseline)
        row = {
            "benchmark": run.benchmark,
            "baseline": run.baseline,
            "run_id": run.run_id,
            "metric": metric,
            "score": score,
            "sample_count": getattr(run, "sample_count", None),
            "parameters": getattr(run, "parameters", {}),
        }
        if key not in best or score > best[key]["score"]:
            best[key] = row
    return sorted(best.values(), key=lambda row: (row["benchmark"], -row["score"], row["baseline"]))


def official_score(metrics: dict[str, Any], metric: str = "acc") -> float | None:
    candidates = {
        "acc": ("overall_acc", "avg_acc", "acc"),
        "avg_acc": ("avg_acc", "overall_acc"),
        "f1": ("overall_f1", "f1"),
    }.get(metric, (metric,))
    for key in candidates:
        value = metrics.get(key)
        if isinstance(value, int | float):
            return float(value)
    return None


def retrieval_diagnostics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        metadata = record.get("prepare_metadata") or {}
        candidates = _retrieval_candidates(metadata)
        evidence_pages = _page_number_set(record.get("evidence_pages") or [])
        hit_candidate = None
        for candidate in candidates:
            if candidate["pages"] & evidence_pages:
                hit_candidate = candidate
                break
        rows.append(
            {
                "question_id": record.get("question_id"),
                "score": display_score(record),
                "evidence_pages": sorted(evidence_pages),
                "evidence_hit": hit_candidate is not None,
                "first_hit_rank": hit_candidate["rank"] if hit_candidate else None,
                "first_hit_score": hit_candidate["score"] if hit_candidate else None,
                "prepare_duration_seconds": _duration(metadata),
                "generation_duration_seconds": _duration(record.get("generation_metadata") or {}),
                "extraction_duration_seconds": _duration(record.get("extraction_metadata") or {}),
                "total_duration_seconds": _duration(metadata)
                + _duration(record.get("generation_metadata") or {})
                + _duration(record.get("extraction_metadata") or {}),
            }
        )
    return rows


def correction_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        metadata = record.get("correction_metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        initial_score = _optional_float(metadata.get("initial_score"))
        corrected_score = _optional_float(metadata.get("corrected_score"))
        final_score = _optional_float(record.get("score"))
        initial_pred = metadata.get("initial_pred")
        corrected_pred = metadata.get("corrected_pred")
        if corrected_pred is None:
            corrected_pred = record.get("pred")
        rows.append(
            {
                "question_id": record.get("question_id"),
                "question": record.get("question"),
                "answer": record.get("answer"),
                "answer_format": record.get("answer_format"),
                "final_pred": record.get("pred"),
                "final_score": final_score,
                "correction_ran": bool(metadata),
                "correction_applied": bool(metadata.get("applied")) if metadata else False,
                "correction_outcome": _correction_outcome(metadata, initial_score, corrected_score, final_score),
                "initial_pred": initial_pred,
                "initial_pred_format": metadata.get("initial_pred_format"),
                "initial_score": initial_score,
                "corrected_pred": corrected_pred,
                "corrected_pred_format": metadata.get("corrected_pred_format"),
                "corrected_score": corrected_score,
                "score_delta": _score_delta(initial_score, corrected_score),
                "pred_changed": _pred_changed(initial_pred, corrected_pred),
                "error": metadata.get("error"),
                "duration_seconds": _optional_float(metadata.get("duration_seconds")),
                "model": metadata.get("model"),
                "corrected_extracted_res": metadata.get("corrected_extracted_res"),
                "input_messages": metadata.get("input_messages") or [],
            }
        )
    return rows


def pairwise_comparison(
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
) -> dict[str, Any]:
    by_id_a = {row.get("question_id"): row for row in records_a if row.get("question_id") is not None}
    by_id_b = {row.get("question_id"): row for row in records_b if row.get("question_id") is not None}
    rows = []
    summary = {"a_wins": 0, "b_wins": 0, "ties": 0, "aligned": 0}
    for question_id in sorted(by_id_a.keys() & by_id_b.keys()):
        score_a = _to_float(by_id_a[question_id].get("score"))
        score_b = _to_float(by_id_b[question_id].get("score"))
        delta = score_a - score_b
        if delta > 0:
            outcome = "A wins"
            summary["a_wins"] += 1
        elif delta < 0:
            outcome = "B wins"
            summary["b_wins"] += 1
        else:
            outcome = "Tie"
            summary["ties"] += 1
        summary["aligned"] += 1
        rows.append(
            {
                "question_id": question_id,
                "score_a": score_a,
                "score_b": score_b,
                "delta": delta,
                "outcome": outcome,
                "question": by_id_a[question_id].get("question") or by_id_b[question_id].get("question"),
                "answer": by_id_a[question_id].get("answer") or by_id_b[question_id].get("answer"),
                "pred_a": by_id_a[question_id].get("pred"),
                "pred_b": by_id_b[question_id].get("pred"),
            }
        )
    rows.sort(key=lambda row: abs(row["delta"]), reverse=True)
    return {"summary": summary, "rows": rows}


def score_bucket(score: Any) -> str:
    value = _to_float(score)
    if value >= 0.999:
        return "correct"
    if value <= 0.001:
        return "wrong"
    return "partial"


def case_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    diagnostics = {row["question_id"]: row for row in retrieval_diagnostics(records)}
    corrections = {row["question_id"]: row for row in correction_rows(records)}
    rows = []
    for record in records:
        question_id = record.get("question_id")
        diagnostic = diagnostics.get(question_id, {})
        correction = corrections.get(question_id, {})
        rows.append(
            {
                "question_id": question_id,
                "question": record.get("question"),
                "answer": record.get("answer"),
                "pred": record.get("pred"),
                "score": display_score(record),
                "score_bucket": score_bucket(display_score(record)),
                "task_tag": record.get("task_tag"),
                "doc_type": record.get("doc_type"),
                "answer_format": record.get("answer_format"),
                "evidence_sources": ", ".join(map(str, record.get("evidence_sources") or [])),
                "evidence_pages": record.get("evidence_pages"),
                "images": record.get("images") or [],
                "metadata": record.get("prepare_metadata") or {},
                "correction_outcome": correction.get("correction_outcome"),
                "correction_applied": correction.get("correction_applied"),
                "initial_score": correction.get("initial_score"),
                "corrected_score": correction.get("corrected_score"),
                "correction_score_delta": correction.get("score_delta"),
                **diagnostic,
            }
        )
    return rows


def aggregate_retrieval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "hit_rate": None, "avg_first_hit_rank": None}
    hit_rows = [row for row in rows if row.get("evidence_hit")]
    ranks = [row["first_hit_rank"] for row in hit_rows if row.get("first_hit_rank") is not None]
    return {
        "count": len(rows),
        "hit_rate": len(hit_rows) / len(rows),
        "avg_first_hit_rank": sum(ranks) / len(ranks) if ranks else None,
    }


def parameter_curve_rows(runs: list[Any], metric: str = "acc") -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        score = official_score(getattr(run, "metrics", {}) or {}, metric)
        if score is None:
            continue
        plugin = get_plugin(getattr(run, "baseline", ""))
        row = {
            "benchmark": run.benchmark,
            "baseline": run.baseline,
            "run_id": run.run_id,
            "score": score,
            "plugin_name": plugin.name,
            "chart_specs": [spec.as_dict() for spec in plugin.chart_specs()],
        }
        row.update(getattr(run, "parameters", {}) or {})
        rows.append(row)
    return rows


def _flatten_node(
    node: Any,
    parts: list[str],
    rows: list[dict[str, Any]],
    inherited_count: int | None,
) -> None:
    if isinstance(node, dict):
        count = node.get("count", inherited_count)
        numeric_items = {
            key: value
            for key, value in node.items()
            if isinstance(value, int | float) and key != "count"
        }
        nested_items = {
            key: value
            for key, value in node.items()
            if isinstance(value, dict)
        }
        for key, value in numeric_items.items():
            path = ".".join([*parts, key])
            rows.append(
                {
                    "path": path,
                    "category": parts[0] if parts else key,
                    "name": key,
                    "value": float(value),
                    "count": count,
                }
            )
        for key, value in nested_items.items():
            _flatten_node(value, [*parts, key], rows, count)


def _retrieval_candidates(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    raw_candidates = []
    for key in ("retrieved_chunks", "retrieved_pages"):
        values = metadata.get(key)
        if isinstance(values, list):
            raw_candidates.extend(values)
    retrieval = metadata.get("retrieval")
    if isinstance(retrieval, dict):
        values = retrieval.get("retrieved_items")
        if isinstance(values, list):
            raw_candidates.extend(values)
        for key in ("initial_retrieved_pages", "final_context_pages"):
            pages = retrieval.get(key)
            if isinstance(pages, list):
                raw_candidates.extend({"rank": index, "page_index": page} for index, page in enumerate(pages, start=1))
    trace = metadata.get("iteration_trace")
    if isinstance(trace, list):
        for step in trace:
            if not isinstance(step, dict):
                continue
            for key in ("retrieved_chunks", "retrieved_pages"):
                values = step.get(key)
                if isinstance(values, list):
                    raw_candidates.extend(values)

    candidates = []
    for index, item in enumerate(raw_candidates, start=1):
        if not isinstance(item, dict):
            continue
        rank = item.get("rank") if isinstance(item.get("rank"), int) else index
        pages = set()
        page_number = item.get("page_number")
        page_index = item.get("page_index")
        if isinstance(page_number, int):
            pages.add(page_number)
        if isinstance(page_index, int):
            pages.add(page_index)
            pages.add(page_index + 1)
        candidates.append({"rank": rank, "score": _to_float(item.get("score")), "pages": pages})
    return sorted(candidates, key=lambda candidate: candidate["rank"])


def _page_number_set(values: Any) -> set[int]:
    if isinstance(values, str):
        text = values.strip()
        if not text:
            values = []
        else:
            try:
                values = ast.literal_eval(text)
            except Exception:
                values = [text]
    if isinstance(values, dict):
        values = list(values.values())
    if not isinstance(values, (list, tuple, set)):
        values = [values]
    pages = set()
    for value in values:
        if isinstance(value, dict):
            value = value.get("page_number", value.get("page", value.get("page_index")))
        if isinstance(value, int):
            pages.add(value)
        elif isinstance(value, str) and value.isdigit():
            pages.add(int(value))
    return pages


def _duration(metadata: dict[str, Any]) -> float:
    return _to_float(metadata.get("duration_seconds"))


def _correction_outcome(
    metadata: dict[str, Any],
    initial_score: float | None,
    corrected_score: float | None,
    final_score: float | None,
) -> str:
    if not metadata:
        return "not_run"
    if metadata.get("error"):
        return "correction_error"
    if _is_correct(initial_score):
        return "skipped_initial_correct"
    if initial_score is None:
        return "not_run"
    if corrected_score is None:
        return "correction_error"
    if _is_wrong(initial_score) and _is_correct(corrected_score):
        return "wrong_to_correct"
    if _is_wrong(initial_score) and _is_wrong(corrected_score):
        return "wrong_to_wrong"
    if corrected_score > initial_score and not _is_correct(corrected_score):
        return "improved_partial"
    if corrected_score < initial_score:
        return "regressed"
    if final_score is not None and corrected_score != final_score:
        return "not_applied"
    return "unchanged"


def _score_delta(initial_score: float | None, corrected_score: float | None) -> float | None:
    if initial_score is None or corrected_score is None:
        return None
    return corrected_score - initial_score


def _pred_changed(initial_pred: Any, corrected_pred: Any) -> bool:
    if initial_pred is None or corrected_pred is None:
        return False
    return str(initial_pred) != str(corrected_pred)


def _is_correct(score: float | None) -> bool:
    return score is not None and score >= 0.999


def _is_wrong(score: float | None) -> bool:
    return score is not None and score <= 0.001


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
