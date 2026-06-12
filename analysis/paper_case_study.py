from __future__ import annotations

import argparse
import json
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd

from analysis.result_fields import display_score


@dataclass(frozen=True)
class CaseStudyArtifacts:
    csv_path: Path
    figure_path: Path
    tex_path: Path


def magerag_case_rows(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [_case_candidate(record) for record in records]
    candidates = [row for row in candidates if row is not None]
    success = _best_case(candidates, "success")
    failure = _best_case(candidates, "failure")
    return [row for row in (success, failure) if row is not None]


def write_case_study_artifacts(
    jsonl_path: Path | str,
    output_dir: Path | str,
    paper_image_dir: Path | str,
    paper_table_dir: Path | str | None = None,
) -> CaseStudyArtifacts:
    output = Path(output_dir)
    image_dir = Path(paper_image_dir)
    table_dir = Path(paper_table_dir) if paper_table_dir is not None else output
    output.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    rows = magerag_case_rows(_read_jsonl(Path(jsonl_path)))
    csv_path = output / "magerag_case_study.csv"
    figure_path = image_dir / "magerag_case_study.pdf"
    tex_path = table_dir / "magerag_case_study_figure.tex"

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    _plot_cases(rows, figure_path)
    tex_path.write_text(_latex_figure(), encoding="utf-8")
    return CaseStudyArtifacts(csv_path=csv_path, figure_path=figure_path, tex_path=tex_path)


def _case_candidate(record: dict[str, Any]) -> dict[str, Any] | None:
    metadata = record.get("prepare_metadata") if isinstance(record.get("prepare_metadata"), dict) else {}
    trace = _trace(metadata)
    if not trace:
        return None
    actions = Counter(str(step.get("action") or "") for step in trace)
    opened_nodes = len(metadata.get("opened_node_ids") or [])
    active_nodes = len(metadata.get("active_node_ids") or [])
    pruned_nodes = len(metadata.get("pruned_node_ids") or [])
    reader_input = metadata.get("reader_input") if isinstance(metadata.get("reader_input"), dict) else {}
    score = _score(record)
    case_type = "success" if score >= 0.999 else "failure"
    search_count = _action_count(actions, ("Search", "FocusedSearch", "SearchNode", "Retrieve", "SearchEvidence", "SearchEvidenceRetrieval"))
    richness = (
        actions.get("EvaluatorDecision", 0) * 5
        + actions.get("OpenNode", 0) * 3
        + actions.get("ActivateNode", 0)
        + search_count * 4
        + opened_nodes
    )
    if richness <= 0:
        return None
    return {
        "case_type": case_type,
        "question_id": record.get("question_id"),
        "question": _compact(record.get("question"), 180),
        "answer": _compact(record.get("answer"), 120),
        "pred": _compact(_prediction(record), 120),
        "score": score,
        "answerable": not _is_unanswerable(record.get("answer")),
        "stop_reason": str(metadata.get("stop_reason") or "unknown"),
        "trace_steps": len(trace),
        "max_iteration": _max_iteration(trace),
        "active_nodes": active_nodes,
        "opened_nodes": opened_nodes,
        "pruned_nodes": pruned_nodes,
        "reader_images": len(reader_input.get("image_refs") or []),
        "content_part_count": _optional_int(reader_input.get("content_part_count")),
        "activate_page_count": actions.get("ActivatePage", 0),
        "activate_node_count": actions.get("ActivateNode", 0),
        "open_node_count": actions.get("OpenNode", 0),
        "evaluator_decisions": actions.get("EvaluatorDecision", 0),
        "search_count": search_count,
        "prune_count": _action_count(actions, ("PruneNode", "PruneNodes", "Prune")),
        "trace_excerpt": _trace_excerpt(trace),
        "page_excerpt": _page_excerpt(trace, reader_input),
        "richness": richness,
    }


def _best_case(candidates: list[dict[str, Any]], case_type: str) -> dict[str, Any] | None:
    subset = [row for row in candidates if row["case_type"] == case_type]
    if not subset:
        return None
    return sorted(
        subset,
        key=lambda row: (
            int(bool(row["answerable"])),
            row["richness"],
            row["opened_nodes"],
            row["search_count"],
            row["trace_steps"],
            str(row.get("question_id") or ""),
        ),
        reverse=True,
    )[0]


def _plot_cases(rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(max(len(rows), 1), 1, figsize=(10.5, 3.6 * max(len(rows), 1)), constrained_layout=True)
    axes = list(getattr(axes, "flat", [axes]))
    for axis, row in zip(axes, rows, strict=False):
        axis.axis("off")
        title = "Success case" if row["case_type"] == "success" else "Failure case"
        lines = [
            f"{title}: {row['question_id']}  |  score={row['score']:.1f}  |  stop={row['stop_reason']}",
            f"Q: {row['question']}",
            f"Gold: {row['answer']}",
            f"Pred: {row['pred']}",
            (
                "Trace: "
                f"steps={row['trace_steps']}, iter={row['max_iteration']}, opened={row['opened_nodes']}, "
                f"active={row['active_nodes']}, pruned={row['pruned_nodes']}, search={row['search_count']}, "
                f"reader images={row['reader_images']}"
            ),
            f"Path: {row['trace_excerpt']}",
            f"Pages: {row['page_excerpt']}",
        ]
        wrapped = "\n".join(textwrap.fill(line, width=132, subsequent_indent="   ") for line in lines)
        axis.text(0.01, 0.98, wrapped, ha="left", va="top", family="monospace", fontsize=8.5)
    for axis in axes[len(rows):]:
        axis.axis("off")
    fig.savefig(path)
    plt.close(fig)


def _latex_figure() -> str:
    return r"""\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{images/magerag_case_study.pdf}
\caption{MAGE-RAG 查询时 evidence subgraph construction 的成功与失败案例。每个案例列出问题、标准答案、预测答案、停止原因、trace 规模、关键动作序列和最终 reader 输入页面摘要。}
\label{fig:magerag_case_study}
\end{figure*}
"""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _trace(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    trace = metadata.get("iteration_trace")
    return [step for step in trace if isinstance(step, dict)] if isinstance(trace, list) else []


def _score(record: dict[str, Any]) -> float:
    return _optional_float(display_score(record)) or 0.0


def _prediction(record: dict[str, Any]) -> Any:
    if record.get("corrected_pred") is not None:
        return record.get("corrected_pred")
    return record.get("pred")


def _trace_excerpt(trace: list[dict[str, Any]], limit: int = 10) -> str:
    actions = [str(step.get("action") or "") for step in trace if step.get("action")]
    excerpt = actions[:limit]
    suffix = " -> ..." if len(actions) > limit else ""
    return " -> ".join(excerpt) + suffix


def _page_excerpt(trace: list[dict[str, Any]], reader_input: dict[str, Any]) -> str:
    pages: list[int] = []
    for step in trace:
        payload = step.get("payload") if isinstance(step.get("payload"), dict) else {}
        page_index = _optional_int(payload.get("page_index"))
        if page_index is not None:
            pages.append(page_index + 1)
    for ref in reader_input.get("image_refs") or []:
        if not isinstance(ref, dict):
            continue
        page_index = _optional_int(ref.get("page_index"))
        if page_index is not None:
            pages.append(page_index + 1)
    unique = list(dict.fromkeys(pages))[:12]
    return ", ".join(str(page) for page in unique) if unique else "n/a"


def _max_iteration(trace: list[dict[str, Any]]) -> int | None:
    values = [_optional_int(step.get("iteration")) for step in trace]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def _action_count(actions: Counter[str], names: tuple[str, ...]) -> int:
    return sum(actions.get(name, 0) for name in names)


def _compact(value: Any, limit: int) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _is_unanswerable(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"not answerable", "unanswerable", "n/a"}
    if isinstance(value, list):
        return any(_is_unanswerable(item) for item in value)
    return False


def _optional_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MAGE-RAG paper case-study artifacts from a JSONL run.")
    parser.add_argument(
        "--jsonl-path",
        type=Path,
        default=Path(
            "results/mmlongbench/magerag/"
            "res_top_k_5_mode_full_watchdog_iterations_10_max_selected_actions_per_iteration_5_mode_full_graph_Qwen3_VL_8B_Instruct.jsonl"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_cache/paper"))
    parser.add_argument("--paper-image-dir", type=Path, default=Path("../paper/images"))
    parser.add_argument("--paper-table-dir", type=Path, default=Path("../paper/tables"))
    args = parser.parse_args()

    artifacts = write_case_study_artifacts(args.jsonl_path, args.output_dir, args.paper_image_dir, args.paper_table_dir)
    for path in artifacts.__dict__.values():
        print(path)


if __name__ == "__main__":
    main()
