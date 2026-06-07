from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from analysis.plugins.base import AnalysisPlugin, ParameterSpec


STATE_RANK = {"Pruned": 4, "Opened": 3, "Active": 2, "Inactive": 1}
STATE_COLORS = {
    "Opened": "#0891b2",
    "Active": "#2563eb",
    "Pruned": "#dc2626",
    "Inactive": "#94a3b8",
}


class MAGERAGPlugin(AnalysisPlugin):
    name = "magerag"
    version = "1"
    baseline_names = ("magerag",)

    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec("top_k", "top-k pages", "int", numeric=True),
            ParameterSpec("evaluator_model_name", "evaluator model", "str"),
        )

    def has_case_visualization(self) -> bool:
        return True

    def diagnostic_rows(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        for record in records:
            metadata = _metadata(record)
            trace = _trace(metadata)
            actions = Counter(str(step.get("action") or "") for step in trace if isinstance(step, dict))
            rows.append(
                {
                    "question_id": record.get("question_id"),
                    "score": _to_float(record.get("score")),
                    "stop_reason": metadata.get("stop_reason"),
                    "trace_steps": len(trace),
                    "opened_nodes": len(metadata.get("opened_node_ids") or []),
                    "active_nodes": len(metadata.get("active_node_ids") or []),
                    "pruned_nodes": len(metadata.get("pruned_node_ids") or []),
                    "validation_errors": len(metadata.get("validation_errors") or []),
                    "activate_page": actions.get("ActivatePage", 0),
                    "activate_node": actions.get("ActivateNode", 0),
                    "open_node": actions.get("OpenNode", 0),
                    "evaluator_decisions": actions.get("EvaluatorDecision", 0),
                    "duration_seconds": _to_float(metadata.get("duration_seconds")),
                }
            )
        return rows

    def case_visualization(self, record: dict[str, Any]) -> dict[str, Any]:
        metadata = _metadata(record)
        graph = _load_graph(metadata.get("graph_dir"))
        trace = _trace(metadata)
        node_states = _node_states(metadata)
        page_lookup = _page_lookup(metadata, graph, node_states, trace)
        trace_rows = _trace_rows(trace, page_lookup)
        action_counts = [
            {"action": action, "count": count}
            for action, count in Counter(row["action"] for row in trace_rows).most_common()
        ]
        page_rows = _page_rows(record, metadata, graph, node_states, trace_rows)
        node_rows = _node_rows(graph, node_states, trace_rows)
        return {
            "plugin_name": self.name,
            "graph_available": graph["available"],
            "summary": _summary(record, metadata, trace, node_states),
            "action_counts": action_counts,
            "page_rows": page_rows,
            "node_rows": node_rows,
            "trace_rows": trace_rows,
            "reader_input": _reader_input(metadata),
            "reader_image_refs": _reader_image_refs(metadata),
            "evaluator_rows": _evaluator_rows(trace_rows),
            "expansion_rows": _expansion_rows(trace_rows),
            "validation_errors": metadata.get("validation_errors") or [],
            "summary_artifacts": metadata.get("summary_artifacts") or [],
        }


PLUGIN = MAGERAGPlugin()


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("prepare_metadata") or {}
    return metadata if isinstance(metadata, dict) else {}


def _trace(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    trace = metadata.get("iteration_trace")
    return [step for step in trace if isinstance(step, dict)] if isinstance(trace, list) else []


def _node_states(metadata: dict[str, Any]) -> dict[str, str]:
    states = metadata.get("final_node_states")
    if isinstance(states, dict):
        return {str(node_id): str(state) for node_id, state in states.items()}
    result = {}
    for key, state in (
        ("active_node_ids", "Active"),
        ("opened_node_ids", "Opened"),
        ("pruned_node_ids", "Pruned"),
    ):
        values = metadata.get(key)
        if isinstance(values, list):
            for node_id in values:
                result[str(node_id)] = state
    return result


def _load_graph(graph_dir: Any) -> dict[str, Any]:
    path = Path(str(graph_dir)) if graph_dir else None
    if not path or not path.exists():
        return {"available": False, "nodes": {}, "edges": {}}
    try:
        nodes = {str(node["id"]): node for node in _read_jsonl(path / "nodes.jsonl") if isinstance(node, dict) and "id" in node}
        edges = {str(edge["id"]): edge for edge in _read_jsonl(path / "edges.jsonl") if isinstance(edge, dict) and "id" in edge}
    except (OSError, json.JSONDecodeError):
        return {"available": False, "nodes": {}, "edges": {}}
    return {"available": True, "nodes": nodes, "edges": edges}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _page_lookup(
    metadata: dict[str, Any],
    graph: dict[str, Any],
    node_states: dict[str, str],
    trace: list[dict[str, Any]],
) -> dict[str, int]:
    lookup = {}
    for node_id, node in graph["nodes"].items():
        page_index = _node_page_index(node)
        if page_index is not None:
            lookup[node_id] = page_index
    for node_id in node_states:
        lookup.setdefault(node_id, _page_from_node_id(node_id))
    for step in trace:
        payload = step.get("payload")
        if not isinstance(payload, dict):
            continue
        node_id = payload.get("node_id") or payload.get("target_id")
        page_index = _optional_int(payload.get("page_index"))
        if node_id is not None and page_index is not None:
            lookup[str(node_id)] = page_index
    for page_index in metadata.get("allowed_pages") or []:
        page_index = _optional_int(page_index)
        if page_index is not None:
            lookup.setdefault(f"page:{page_index}", page_index)
    return {key: value for key, value in lookup.items() if value is not None}


def _summary(
    record: dict[str, Any],
    metadata: dict[str, Any],
    trace: list[dict[str, Any]],
    node_states: dict[str, str],
) -> dict[str, Any]:
    state_counts = Counter(node_states.values())
    return {
        "question_id": record.get("question_id"),
        "score": _to_float(record.get("score")),
        "stop_reason": metadata.get("stop_reason"),
        "trace_steps": len(trace),
        "opened_nodes": state_counts.get("Opened", len(metadata.get("opened_node_ids") or [])),
        "active_nodes": state_counts.get("Active", len(metadata.get("active_node_ids") or [])),
        "pruned_nodes": state_counts.get("Pruned", len(metadata.get("pruned_node_ids") or [])),
        "validation_errors": len(metadata.get("validation_errors") or []),
        "duration_seconds": _to_float(metadata.get("duration_seconds")),
        "evaluator_model_name": metadata.get("evaluator_model_name"),
    }


def _trace_rows(trace: list[dict[str, Any]], page_lookup: dict[str, int]) -> list[dict[str, Any]]:
    rows = []
    for index, step in enumerate(trace):
        payload = step.get("payload") if isinstance(step.get("payload"), dict) else {}
        node_id = _payload_node_id(payload)
        page_index = _optional_int(payload.get("page_index"))
        if page_index is None and node_id:
            page_index = page_lookup.get(node_id) or _page_from_node_id(node_id)
        rows.append(
            {
                "step_index": index,
                "iteration": step.get("iteration"),
                "action": str(step.get("action") or ""),
                "ok": bool(step.get("ok", True)),
                "message": step.get("message") or "",
                "node_id": node_id,
                "edge_id": payload.get("edge_id"),
                "page_index": page_index,
                "page_number": page_index + 1 if page_index is not None else None,
                "state_delta": _state_delta(step, payload),
                "payload": payload,
                "decision": step.get("decision") if isinstance(step.get("decision"), dict) else None,
                "raw_response": step.get("raw_response"),
                "evaluator_input": step.get("evaluator_input") if isinstance(step.get("evaluator_input"), dict) else None,
                "selection": step.get("selection") if isinstance(step.get("selection"), dict) else None,
                "state_snapshot_before": step.get("state_snapshot_before") if isinstance(step.get("state_snapshot_before"), dict) else None,
                "state_snapshot_after": step.get("state_snapshot_after") if isinstance(step.get("state_snapshot_after"), dict) else None,
            }
        )
    return rows


def _reader_input(metadata: dict[str, Any]) -> dict[str, Any]:
    value = metadata.get("reader_input")
    return value if isinstance(value, dict) else {}


def _reader_image_refs(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    refs = _reader_input(metadata).get("image_refs")
    return [ref for ref in refs if isinstance(ref, dict)] if isinstance(refs, list) else []


def _evaluator_rows(trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in trace_rows:
        evaluator_input = row.get("evaluator_input")
        if row.get("action") != "EvaluatorDecision" and not evaluator_input and not row.get("raw_response"):
            continue
        evaluator_input = evaluator_input if isinstance(evaluator_input, dict) else {}
        candidate_actions = evaluator_input.get("candidate_actions")
        opened_image_refs = evaluator_input.get("opened_image_refs")
        rows.append({
            "step_index": row.get("step_index"),
            "iteration": row.get("iteration"),
            "prompt_text": evaluator_input.get("prompt_text"),
            "context_xml": evaluator_input.get("context_xml"),
            "candidate_actions": candidate_actions if isinstance(candidate_actions, list) else [],
            "candidate_action_count": len(candidate_actions) if isinstance(candidate_actions, list) else 0,
            "opened_image_refs": opened_image_refs if isinstance(opened_image_refs, list) else [],
            "raw_response": row.get("raw_response"),
            "decision": row.get("decision"),
            "state_snapshot_before": row.get("state_snapshot_before"),
        })
    return rows


def _expansion_rows(trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in trace_rows:
        before = row.get("state_snapshot_before") if isinstance(row.get("state_snapshot_before"), dict) else {}
        after = row.get("state_snapshot_after") if isinstance(row.get("state_snapshot_after"), dict) else before
        selection = row.get("selection") if isinstance(row.get("selection"), dict) else {}
        rows.append({
            "step_index": row.get("step_index"),
            "iteration": row.get("iteration"),
            "action": row.get("action"),
            "ok": row.get("ok"),
            "page_index": row.get("page_index"),
            "page_number": row.get("page_number"),
            "node_id": row.get("node_id"),
            "edge_id": row.get("edge_id"),
            "message": row.get("message"),
            "selected_candidate_index": selection.get("candidate_index"),
            "selected_candidate_id": selection.get("candidate_id") or selection.get("resolved_candidate_id"),
            "selected_candidate_type": selection.get("candidate_action_type"),
            "selection_reason": selection.get("reason"),
            "active_nodes": _snapshot_count(after, "active"),
            "opened_nodes": _snapshot_count(after, "opened"),
            "pruned_nodes": _snapshot_count(after, "pruned"),
            "active_delta": _snapshot_delta(before, after, "active_node_ids"),
            "opened_delta": _snapshot_delta(before, after, "opened_node_ids"),
            "pruned_delta": _snapshot_delta(before, after, "pruned_node_ids"),
        })
    return rows


def _snapshot_count(snapshot: dict[str, Any], state: str) -> int | None:
    count = snapshot.get(f"{state}_count")
    if count is not None:
        return _optional_int(count)
    values = snapshot.get(f"{state}_node_ids")
    return len(values) if isinstance(values, list) else None


def _snapshot_delta(before: dict[str, Any], after: dict[str, Any], key: str) -> list[str]:
    before_values = set(str(value) for value in before.get(key) or []) if isinstance(before.get(key), list) else set()
    after_values = set(str(value) for value in after.get(key) or []) if isinstance(after.get(key), list) else set()
    return sorted(after_values - before_values)


def _page_rows(
    record: dict[str, Any],
    metadata: dict[str, Any],
    graph: dict[str, Any],
    node_states: dict[str, str],
    trace_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pages: dict[int, dict[str, Any]] = {}
    for page_index in metadata.get("allowed_pages") or []:
        _ensure_page(pages, _optional_int(page_index))
    for node_id, node in graph["nodes"].items():
        page_index = _node_page_index(node)
        page = _ensure_page(pages, page_index)
        if page is not None and _is_page_node(node):
            page["page_node_id"] = node_id
            page["page_image_path"] = _image_path(node)
            page["page_preview"] = _preview(node)
    for node_id, state in node_states.items():
        page_index = _node_page_index(graph["nodes"].get(node_id, {}))
        if page_index is None:
            page_index = _page_from_node_id(node_id)
        page = _ensure_page(pages, page_index)
        if page is None:
            continue
        page[f"{state.lower()}_nodes"] += 1
        if STATE_RANK.get(state, 0) > STATE_RANK.get(page["dominant_state"], 0):
            page["dominant_state"] = state
    for row in trace_rows:
        page = _ensure_page(pages, row.get("page_index"))
        if page is not None:
            page["action_count"] += 1
    for page_index in _evidence_page_candidates(record.get("evidence_pages") or []):
        page = _ensure_page(pages, page_index)
        if page is not None:
            page["is_evidence_page"] = True
    for retrieved in _retrieved_pages(metadata):
        page_index = _optional_int(retrieved.get("page_index"))
        if page_index is None and _optional_int(retrieved.get("page_number")) is not None:
            page_index = int(retrieved["page_number"]) - 1
        page = _ensure_page(pages, page_index)
        if page is None:
            continue
        page["retrieval_rank"] = retrieved.get("rank")
        page["retrieval_score"] = _to_float(retrieved.get("score"))
    for page in pages.values():
        page["color"] = STATE_COLORS.get(page["dominant_state"], STATE_COLORS["Inactive"])
    return [pages[index] for index in sorted(pages)]


def _ensure_page(pages: dict[int, dict[str, Any]], page_index: Any) -> dict[str, Any] | None:
    page_index = _optional_int(page_index)
    if page_index is None:
        return None
    if page_index not in pages:
        pages[page_index] = {
            "page_index": page_index,
            "page_number": page_index + 1,
            "dominant_state": "Inactive",
            "color": STATE_COLORS["Inactive"],
            "active_nodes": 0,
            "opened_nodes": 0,
            "pruned_nodes": 0,
            "action_count": 0,
            "is_evidence_page": False,
            "retrieval_rank": None,
            "retrieval_score": None,
            "page_image_path": None,
            "page_preview": "",
            "page_node_id": None,
        }
    page = pages[page_index]
    page["color"] = STATE_COLORS.get(page["dominant_state"], STATE_COLORS["Inactive"])
    return page


def _node_rows(graph: dict[str, Any], node_states: dict[str, str], trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    action_counts = Counter(row.get("node_id") for row in trace_rows if row.get("node_id"))
    rows = []
    for node_id, state in sorted(node_states.items()):
        node = graph["nodes"].get(node_id, {})
        if _is_page_node(node):
            continue
        rows.append(
            {
                "node_id": node_id,
                "state": state,
                "type": node.get("type"),
                "page_index": _node_page_index(node) if node else _page_from_node_id(node_id),
                "page_number": (_node_page_index(node) + 1) if node and _node_page_index(node) is not None else None,
                "preview": _preview(node),
                "image_path": _image_path(node),
                "bbox": node.get("bbox"),
                "action_count": action_counts.get(node_id, 0),
            }
        )
    return rows


def _retrieved_pages(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    initial = metadata.get("initial_retrieval")
    if not isinstance(initial, dict):
        return []
    pages = initial.get("retrieved_pages")
    if isinstance(pages, list) and pages:
        return [page for page in pages if isinstance(page, dict)]
    initial_page = initial.get("initial_page")
    return [initial_page] if isinstance(initial_page, dict) else []


def _payload_node_id(payload: dict[str, Any]) -> str | None:
    for key in ("node_id", "target_id"):
        value = payload.get(key)
        if value is not None:
            return str(value)
    return None


def _state_delta(step: dict[str, Any], payload: dict[str, Any]) -> str:
    previous = payload.get("previous_state")
    action = str(step.get("action") or "")
    target = {
        "ActivatePage": "Active",
        "ActivateNode": "Active",
        "OpenNode": "Opened",
        "PruneNode": "Pruned",
    }.get(action)
    if previous and target:
        return f"{previous} -> {target}"
    if target:
        return f"-> {target}"
    return ""


def _node_page_index(node: dict[str, Any]) -> int | None:
    return _optional_int(node.get("page_index", node.get("page_no")))


def _page_from_node_id(node_id: str) -> int | None:
    parts = str(node_id).split(":")
    if "page" not in parts:
        return None
    index = parts.index("page")
    if index + 1 >= len(parts):
        return None
    return _optional_int(parts[index + 1])


def _is_page_node(node: dict[str, Any]) -> bool:
    return str(node.get("type") or "").lower() == "page"


def _preview(node: dict[str, Any]) -> str:
    for key in ("abstract", "text", "content", "caption", "html"):
        value = node.get(key)
        if value:
            text = str(value).strip()
            return text[:240] + ("..." if len(text) > 240 else "")
    return ""


def _image_path(node: dict[str, Any]) -> str | None:
    value = node.get("image_path") or node.get("crop_path")
    if not value and isinstance(node.get("metadata"), dict):
        value = node["metadata"].get("image_path") or node["metadata"].get("crop_path")
    return str(value) if value else None


def _evidence_page_candidates(values: list[Any]) -> set[int]:
    pages = set()
    for value in values:
        number = _optional_int(value)
        if number is not None:
            pages.add(number)
            if number > 0:
                pages.add(number - 1)
    return pages


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
