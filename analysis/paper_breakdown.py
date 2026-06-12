from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd

from analysis.paper_main_results import main_result_rows
from analysis.results_loader import scan_runs


BREAKDOWNS = [
    ("LongDocURL", "Task", "Understanding", "longdocurl_understanding"),
    ("LongDocURL", "Task", "Reasoning", "longdocurl_reasoning"),
    ("LongDocURL", "Task", "Locating", "longdocurl_locating"),
    ("LongDocURL", "Element type", "Text", "longdocurl_text"),
    ("LongDocURL", "Element type", "Layout", "longdocurl_layout"),
    ("LongDocURL", "Element type", "Figure", "longdocurl_figure"),
    ("LongDocURL", "Element type", "Table", "longdocurl_table"),
    ("LongDocURL", "Evidence pages", "Single-page", "longdocurl_single_page"),
    ("LongDocURL", "Evidence pages", "Multi-page", "longdocurl_multi_page"),
    ("MMLongBench-Doc", "Question type", "Single-page", "mmlongbench_single_page"),
    ("MMLongBench-Doc", "Question type", "Cross-page", "mmlongbench_cross_page"),
    ("MMLongBench-Doc", "Question type", "Unanswerable", "mmlongbench_unanswerable"),
]


@dataclass(frozen=True)
class BreakdownArtifacts:
    csv_path: Path
    figure_path: Path
    tex_path: Path


def breakdown_rows(main_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for main_row in main_rows:
        method = str(main_row["method"])
        method_type = str(main_row.get("type", ""))
        for benchmark, group, split, key in BREAKDOWNS:
            score = main_row.get(key)
            if score is None or pd.isna(score):
                continue
            score_float = float(score)
            rows.append(
                {
                    "method": method,
                    "type": method_type,
                    "benchmark": benchmark,
                    "group": group,
                    "split": split,
                    "score": score_float,
                    "score_pct": score_float * 100,
                }
            )
    return rows


def write_breakdown_artifacts(
    main_rows: Iterable[dict[str, Any]],
    output_dir: Path | str,
    paper_image_dir: Path | str,
    paper_table_dir: Path | str | None = None,
) -> BreakdownArtifacts:
    output = Path(output_dir)
    image_dir = Path(paper_image_dir)
    table_dir = Path(paper_table_dir) if paper_table_dir is not None else output
    output.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    rows = breakdown_rows(main_rows)
    df = pd.DataFrame(rows, columns=["method", "type", "benchmark", "group", "split", "score", "score_pct"])
    csv_path = output / "main_breakdown.csv"
    figure_path = image_dir / "main_breakdown.pdf"
    tex_path = table_dir / "main_breakdown_figure.tex"

    df.to_csv(csv_path, index=False)
    _plot_breakdowns(df, figure_path)
    tex_path.write_text(_latex_figure(), encoding="utf-8")
    return BreakdownArtifacts(csv_path=csv_path, figure_path=figure_path, tex_path=tex_path)


def _plot_breakdowns(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.1), constrained_layout=True)
    panels = [
        ("LongDocURL", "Task", "LongDocURL task splits"),
        ("LongDocURL", "Evidence pages", "LongDocURL evidence pages"),
        ("LongDocURL", "Element type", "LongDocURL element types"),
        ("MMLongBench-Doc", "Question type", "MMLongBench question types"),
    ]
    colors = plt.get_cmap("tab10")
    method_order = list(dict.fromkeys(df["method"].tolist())) if not df.empty else []

    for axis, (benchmark, group, title) in zip(axes.flat, panels, strict=True):
        subset = df[(df["benchmark"] == benchmark) & (df["group"] == group)].copy()
        _plot_panel(axis, subset, method_order, title, colors)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", frameon=False, ncol=min(len(labels), 4))
    fig.savefig(path)
    plt.close(fig)


def _plot_panel(axis: plt.Axes, df: pd.DataFrame, method_order: list[str], title: str, colors: Any) -> None:
    axis.set_title(title)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Accuracy (%)")
    if df.empty:
        axis.text(0.5, 0.5, "No reproduced metrics", ha="center", va="center", transform=axis.transAxes)
        return

    split_order = list(dict.fromkeys(df["split"].tolist()))
    x_positions = range(len(split_order))
    width = 0.8 / max(len(method_order), 1)
    for method_index, method in enumerate(method_order):
        values = []
        for split in split_order:
            match = df[(df["method"] == method) & (df["split"] == split)]
            values.append(float(match.iloc[0]["score_pct"]) if not match.empty else 0.0)
        offsets = [x - 0.4 + width / 2 + method_index * width for x in x_positions]
        axis.bar(offsets, values, width=width, label=method, color=colors(method_index % 10))
    axis.set_xticks(list(x_positions), split_order, rotation=20, ha="right")
    axis.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)


def _latex_figure() -> str:
    return r"""\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{images/main_breakdown.pdf}
\caption{LongDocURL 和 MMLongBench-Doc 上统一复现实验的分层结果。LongDocURL 按任务类型、evidence-page 数量和 element type 拆分；MMLongBench-Doc 按 single-page、cross-page 和 unanswerable 问题拆分。缺失 metrics 的方法不在对应面板中绘制。}
\label{fig:main_breakdown}
\end{figure*}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper fine-grained breakdown figure from main-result metrics.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_cache/paper"))
    parser.add_argument("--paper-image-dir", type=Path, default=Path("../paper/images"))
    parser.add_argument("--paper-table-dir", type=Path, default=Path("../paper/tables"))
    args = parser.parse_args()

    artifacts = write_breakdown_artifacts(
        main_result_rows(scan_runs(args.results_root)),
        args.output_dir,
        args.paper_image_dir,
        args.paper_table_dir,
    )
    for path in artifacts.__dict__.values():
        print(path)


if __name__ == "__main__":
    main()
