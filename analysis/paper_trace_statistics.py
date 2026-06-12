from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd

from analysis.result_fields import display_score


@dataclass(frozen=True)
class TraceArtifacts:
    trace_csv: Path
    summary_csv: Path
    stop_reason_csv: Path
    summary_tex: Path
    figure: Path


def magerag_trace_rows(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        metadata = record.get("prepare_metadata") or {}
        metadata = metadata if isinstance(metadata, dict) else {}
        trace = _trace(metadata)
        summary = _trace_summary(metadata)
        actions = Counter(str(step.get("action") or "") for step in trace)
        reader_input = metadata.get("reader_input") if isinstance(metadata.get("reader_input"), dict) else {}
        logical_cost = metadata.get("logical_cost") if isinstance(metadata.get("logical_cost"), dict) else {}
        rows.append(
            {
                "question_id": record.get("question_id"),
                "score": _to_float(display_score(record)),
                "correctness": _correctness(display_score(record)),
                "stop_reason": str(summary.get("stop_reason") or metadata.get("stop_reason") or "unknown"),
                "trace_steps": _first_int(summary.get("num_trace_events"), len(trace)),
                "max_iteration": _first_int(summary.get("num_iterations"), _max_iteration(trace)),
                "activated_pages": _first_int(summary.get("num_activate_page"), _activated_pages(metadata, actions)),
                "active_nodes": len(metadata.get("active_node_ids") or []),
                "opened_nodes": _first_int(summary.get("num_open_node"), len(metadata.get("opened_node_ids") or [])),
                "pruned_nodes": _first_int(summary.get("num_prune_node"), len(metadata.get("pruned_node_ids") or [])),
                "activate_page_count": actions.get("ActivatePage", 0),
                "activate_node_count": actions.get("ActivateNode", 0),
                "open_node_count": actions.get("OpenNode", 0),
                "search_count": _first_int(
                    _summary_search_count(summary),
                    _action_count(
                        actions,
                        ("Search", "FocusedSearch", "SearchNode", "Retrieve", "SearchEvidence", "SearchEvidenceRetrieval"),
                    ),
                ),
                "prune_count": _first_int(
                    summary.get("num_prune_node"),
                    _action_count(actions, ("PruneNode", "PruneNodes", "Prune")),
                ),
                "evaluator_decisions": actions.get("EvaluatorDecision", 0),
                "reader_images": len(reader_input.get("image_refs") or []),
                "content_part_count": _optional_int(reader_input.get("content_part_count")),
                "estimated_input_tokens": _optional_float(logical_cost.get("estimated_input_tokens")),
                "llm_calls": _first_float(logical_cost.get("num_llm_calls"), logical_cost.get("llm_calls")),
                "retriever_calls": _first_float(logical_cost.get("num_retriever_calls"), logical_cost.get("retriever_calls")),
            }
        )
    return rows


def write_trace_statistics_artifacts(
    jsonl_path: Path | str,
    output_dir: Path | str,
    paper_image_dir: Path | str,
    paper_table_dir: Path | str | None = None,
) -> TraceArtifacts:
    output = Path(output_dir)
    image_dir = Path(paper_image_dir)
    table_dir = Path(paper_table_dir) if paper_table_dir is not None else output
    output.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    records = _read_jsonl(Path(jsonl_path))
    rows = magerag_trace_rows(records)
    df = pd.DataFrame(rows)
    summary = _summary_table(df)
    stop_reasons = _stop_reason_table(df)

    trace_csv = output / "mmlongbench_magerag_trace_rows.csv"
    summary_csv = output / "mmlongbench_magerag_trace_summary.csv"
    stop_reason_csv = output / "mmlongbench_magerag_stop_reasons.csv"
    summary_tex = table_dir / "mmlongbench_magerag_trace_stats_table.tex"
    figure = image_dir / "mmlongbench_magerag_trace_stats.pdf"

    df.to_csv(trace_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    stop_reasons.to_csv(stop_reason_csv, index=False)
    summary_tex.write_text(_latex_summary_table(summary), encoding="utf-8")
    _plot_trace_statistics(summary, stop_reasons, figure)

    return TraceArtifacts(
        trace_csv=trace_csv,
        summary_csv=summary_csv,
        stop_reason_csv=stop_reason_csv,
        summary_tex=summary_tex,
        figure=figure,
    )


def _summary_table(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "trace_steps",
        "max_iteration",
        "activated_pages",
        "opened_nodes",
        "pruned_nodes",
        "search_count",
        "prune_count",
        "reader_images",
        "content_part_count",
    ]
    rows = []
    for correctness in ("correct", "incorrect"):
        subset = df[df["correctness"] == correctness] if not df.empty else df
        row = {"correctness": correctness, "count": int(len(subset))}
        for metric in metrics:
            row[f"avg_{metric}"] = float(subset[metric].mean()) if len(subset) else None
        rows.append(row)
    return pd.DataFrame(rows)


def _stop_reason_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["correctness", "stop_reason", "count"])
    return (
        df.groupby(["correctness", "stop_reason"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["correctness", "count", "stop_reason"], ascending=[True, False, True])
    )


def _latex_summary_table(df: pd.DataFrame) -> str:
    header = r"""\begin{table}[t]
\centering
\caption{MMLongBench-Doc 上 MAGE-RAG trace 统计。统计基于代表性运行 $k=3,T=5,a=3$；均值按 answer-level correct/incorrect 切分。}
\label{tab:magerag_trace_stats}
\scriptsize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccccccc}
\hline
Group & Count & Steps & Iter. & Pages & Open & Prune & Search & Images \\
\hline
"""
    body = "".join(
        f"{_label(row.correctness)} & {int(row.count)} & {_num(row.avg_trace_steps)} & "
        f"{_num(row.avg_max_iteration)} & {_num(row.avg_activated_pages)} & "
        f"{_num(row.avg_opened_nodes)} & {_num(row.avg_pruned_nodes)} & "
        f"{_num(row.avg_search_count)} & {_num(row.avg_reader_images)} \\\\\n"
        for row in df.itertuples(index=False)
    )
    footer = r"""\hline
\end{tabular}%
}
\end{table}
"""
    return header + body + footer


def _plot_trace_statistics(summary: pd.DataFrame, stop_reasons: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1), constrained_layout=True)
    plot_summary = summary.set_index("correctness")
    panels = [
        (
            axes[0],
            ["avg_opened_nodes", "avg_activated_pages", "avg_search_count", "avg_reader_images"],
            ["Opened nodes", "Activated pages", "Search", "Reader images"],
            "Evidence breadth",
        ),
        (
            axes[1],
            ["avg_trace_steps", "avg_max_iteration", "avg_prune_count"],
            ["Trace steps", "Iterations", "Prune"],
            "Trace length and pruning",
        ),
    ]
    width = 0.36
    for axis, metric_cols, labels, title in panels:
        x = range(len(labels))
        for offset, correctness in [(-width / 2, "correct"), (width / 2, "incorrect")]:
            values = [plot_summary.loc[correctness, col] if correctness in plot_summary.index else 0 for col in metric_cols]
            axis.bar([value + offset for value in x], values, width=width, label=_label(correctness))
        axis.set_xticks(list(x), labels, rotation=20, ha="right")
        axis.set_ylabel("Average count")
        axis.set_title(title)
        axis.grid(axis="y", linewidth=0.4, alpha=0.4)
    axes[0].legend(fontsize=8)
    fig.savefig(path)
    plt.close(fig)


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


def _trace_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    magerag = metadata.get("magerag")
    if isinstance(magerag, dict) and isinstance(magerag.get("trace_summary"), dict):
        return magerag["trace_summary"]
    summary = metadata.get("trace_summary")
    return summary if isinstance(summary, dict) else {}


def _summary_search_count(summary: dict[str, Any]) -> int | None:
    request_count = _optional_int(summary.get("num_search_requests"))
    retrieval_count = _optional_int(summary.get("num_search_retrievals"))
    if request_count is None and retrieval_count is None:
        return None
    return int(request_count or 0) + int(retrieval_count or 0)


def _correctness(score: Any) -> str:
    return "correct" if _to_float(score) >= 0.999 else "incorrect"


def _activated_pages(metadata: dict[str, Any], actions: Counter[str]) -> int:
    pages = metadata.get("activated_pages")
    if isinstance(pages, list):
        return len(set(pages))
    return actions.get("ActivatePage", 0)


def _action_count(actions: Counter[str], names: tuple[str, ...]) -> int:
    return sum(actions.get(name, 0) for name in names)


def _max_iteration(trace: list[dict[str, Any]]) -> int:
    values = [_optional_int(step.get("iteration")) for step in trace]
    values = [value for value in values if value is not None]
    return max(values) if values else 0


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_int(*values: Any) -> int:
    for value in values:
        parsed = _optional_int(value)
        if parsed is not None:
            return parsed
    return 0


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _optional_float(value)
        if parsed is not None:
            return parsed
    return None


def _num(value: Any) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.2f}"


def _label(value: Any) -> str:
    return str(value).replace("_", " ").title()


def _short_reason(reason: Any, correctness: Any) -> str:
    label = str(reason).replace("_", " ")
    return f"{_label(correctness)}: {label[:28]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper artifacts for MAGE-RAG trace statistics.")
    parser.add_argument(
        "--jsonl-path",
        type=Path,
        default=Path(
            "results/mmlongbench/magerag/"
            "res_top_k_3_mode_full_watchdog_iterations_5_max_selected_actions_per_iteration_3_mode_full_graph_Qwen3_VL_8B_Instruct.jsonl"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_cache/paper"))
    parser.add_argument("--paper-image-dir", type=Path, default=Path("../paper/images"))
    parser.add_argument("--paper-table-dir", type=Path, default=Path("../paper/tables"))
    args = parser.parse_args()

    artifacts = write_trace_statistics_artifacts(
        args.jsonl_path,
        args.output_dir,
        args.paper_image_dir,
        args.paper_table_dir,
    )
    for path in artifacts.__dict__.values():
        print(path)


if __name__ == "__main__":
    main()
