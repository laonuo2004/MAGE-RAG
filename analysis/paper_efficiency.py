from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd

from analysis.result_fields import display_score


@dataclass(frozen=True)
class EfficiencyArtifacts:
    rows_csv: Path
    summary_csv: Path
    summary_tex: Path
    figure_path: Path


def magerag_efficiency_rows(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        metadata = record.get("prepare_metadata") if isinstance(record.get("prepare_metadata"), dict) else {}
        reader_input = metadata.get("reader_input") if isinstance(metadata.get("reader_input"), dict) else {}
        logical_cost = metadata.get("logical_cost") if isinstance(metadata.get("logical_cost"), dict) else {}
        score = _optional_float(display_score(record)) or 0.0
        rows.append(
            {
                "question_id": record.get("question_id"),
                "score": score,
                "correctness": "correct" if score >= 0.999 else "incorrect",
                "input_images": _first_int(
                    logical_cost.get("num_input_images"),
                    len(reader_input.get("image_refs") or []),
                ),
                "content_part_count": _first_int(reader_input.get("content_part_count"), len(reader_input.get("messages") or [])),
                "context_pages": _first_int(
                    logical_cost.get("num_context_pages"),
                    _list_len(metadata.get("activated_pages")),
                    _list_len(metadata.get("active_node_ids")),
                ),
                "context_nodes": _first_int(
                    logical_cost.get("num_context_nodes"),
                    _list_len(metadata.get("opened_node_ids")),
                    _list_len(metadata.get("active_node_ids")),
                ),
                "retriever_calls": _first_int(
                    logical_cost.get("num_retriever_calls"),
                    logical_cost.get("retriever_calls"),
                    0,
                ),
                "llm_calls": _first_int(logical_cost.get("num_llm_calls"), logical_cost.get("llm_calls"), 0),
                "input_text_chars": _optional_int(logical_cost.get("num_input_text_chars")),
                "estimated_input_tokens": _optional_float(logical_cost.get("estimated_input_tokens")),
            }
        )
    return rows


def write_efficiency_artifacts(
    jsonl_path: Path | str,
    output_dir: Path | str,
    paper_image_dir: Path | str,
    paper_table_dir: Path | str | None = None,
) -> EfficiencyArtifacts:
    output = Path(output_dir)
    image_dir = Path(paper_image_dir)
    table_dir = Path(paper_table_dir) if paper_table_dir is not None else output
    output.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    rows = magerag_efficiency_rows(_read_jsonl(Path(jsonl_path)))
    df = pd.DataFrame(rows)
    summary = _summary_table(df)

    rows_csv = output / "mmlongbench_magerag_efficiency_rows.csv"
    summary_csv = output / "mmlongbench_magerag_efficiency_summary.csv"
    summary_tex = table_dir / "mmlongbench_magerag_efficiency_table.tex"
    figure_path = image_dir / "mmlongbench_magerag_efficiency.pdf"

    df.to_csv(rows_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_tex.write_text(_latex_summary_table(summary), encoding="utf-8")
    _plot_efficiency(summary, figure_path)
    return EfficiencyArtifacts(rows_csv=rows_csv, summary_csv=summary_csv, summary_tex=summary_tex, figure_path=figure_path)


def _summary_table(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "input_images",
        "context_pages",
        "context_nodes",
        "content_part_count",
        "retriever_calls",
        "llm_calls",
        "input_text_chars",
        "estimated_input_tokens",
    ]
    rows = []
    for group, subset in [("overall", df), ("correct", df[df["correctness"] == "correct"] if not df.empty else df), ("incorrect", df[df["correctness"] == "incorrect"] if not df.empty else df)]:
        score_sum = float(subset["score"].sum()) if len(subset) else 0.0
        image_sum = float(subset["input_images"].sum()) if len(subset) else 0.0
        llm_sum = float(subset["llm_calls"].sum()) if len(subset) else 0.0
        text_sum = float(subset["input_text_chars"].dropna().sum()) if len(subset) else 0.0
        row: dict[str, Any] = {
            "group": group,
            "count": int(len(subset)),
            "acc": float(subset["score"].mean()) if len(subset) else None,
            "score_sum": score_sum,
            "score_per_input_image": score_sum / image_sum if image_sum else None,
            "score_per_llm_call": score_sum / llm_sum if llm_sum else None,
            "score_per_1k_text_chars": score_sum / (text_sum / 1000.0) if text_sum else None,
        }
        for column in columns:
            row[f"avg_{column}"] = float(subset[column].mean()) if len(subset) else None
        rows.append(row)
    return pd.DataFrame(rows)


def _latex_summary_table(df: pd.DataFrame) -> str:
    header = r"""\begin{table}[t]
\centering
\caption{MMLongBench-Doc 上 MAGE-RAG 的 logical cost 统计。统计基于代表性运行 $k=3,T=5,a=3$；Score/Image 和 Score/LLM 为聚合分数除以对应聚合成本。}
\label{tab:magerag_efficiency}
\scriptsize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccccccc}
\hline
Group & Count & Acc & Images & Pages & LLM & Parts & Nodes & Score/Image & Score/LLM \\
\hline
"""
    body = "".join(
        f"{_label(row.group)} & {int(row.count)} & {_pct(row.acc)} & {_num(row.avg_input_images)} & "
        f"{_num(row.avg_context_pages)} & {_num(row.avg_llm_calls)} & {_num(row.avg_content_part_count)} & "
        f"{_num(row.avg_context_nodes)} & {_ratio(row.score_per_input_image)} & {_ratio(row.score_per_llm_call)} \\\\\n"
        for row in df.itertuples(index=False)
    )
    footer = r"""\hline
\end{tabular}%
}
\end{table}
"""
    return header + body + footer


def _plot_efficiency(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1), constrained_layout=True)
    plot_df = summary[summary["group"].isin(["correct", "incorrect"])].set_index("group")
    labels = ["Images", "Pages", "Nodes", "LLM calls"]
    columns = ["avg_input_images", "avg_context_pages", "avg_context_nodes", "avg_llm_calls"]
    x = range(len(labels))
    width = 0.36
    for offset, group in [(-width / 2, "correct"), (width / 2, "incorrect")]:
        values = [plot_df.loc[group, column] if group in plot_df.index else 0 for column in columns]
        axes[0].bar([value + offset for value in x], values, width=width, label=_label(group))
    axes[0].set_xticks(list(x), labels, rotation=20, ha="right")
    axes[0].set_ylabel("Average count")
    axes[0].set_title("Logical cost")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", linewidth=0.4, alpha=0.4)

    overall = summary[summary["group"] == "overall"]
    ratios = overall.iloc[0] if len(overall) else None
    ratio_labels = ["Score/Image", "Score/LLM", "Score/1k chars"]
    ratio_values = [
        _plot_value(ratios, "score_per_input_image"),
        _plot_value(ratios, "score_per_llm_call"),
        _plot_value(ratios, "score_per_1k_text_chars"),
    ]
    axes[1].bar(ratio_labels, ratio_values)
    axes[1].set_title("Aggregate efficiency")
    axes[1].set_ylabel("Score per unit")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", linewidth=0.4, alpha=0.4)
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


def _first_int(*values: Any) -> int | None:
    for value in values:
        parsed = _optional_int(value)
        if parsed is not None:
            return parsed
    return None


def _list_len(value: Any) -> int | None:
    return len(value) if isinstance(value, list) else None


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


def _num(value: Any) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.2f}"


def _pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value) * 100:.2f}"


def _ratio(value: Any) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.3f}"


def _label(value: Any) -> str:
    return str(value).replace("_", " ").title()


def _plot_value(row: Any, key: str) -> float:
    if row is None:
        return 0.0
    value = row.get(key) if hasattr(row, "get") else getattr(row, key, None)
    return 0.0 if value is None or pd.isna(value) else float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MAGE-RAG efficiency paper artifacts from a JSONL run.")
    parser.add_argument("--jsonl-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_cache/paper"))
    parser.add_argument("--paper-image-dir", type=Path, default=Path("../paper/images"))
    parser.add_argument("--paper-table-dir", type=Path, default=Path("../paper/tables"))
    args = parser.parse_args()
    artifacts = write_efficiency_artifacts(args.jsonl_path, args.output_dir, args.paper_image_dir, args.paper_table_dir)
    for path in artifacts.__dict__.values():
        print(path)


if __name__ == "__main__":
    main()
