from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd

from analysis.results_loader import RunRecord, scan_runs


@dataclass(frozen=True)
class PaperArtifacts:
    topk_csv: Path
    budget_csv: Path
    ablation_csv: Path
    topk_tex: Path
    budget_tex: Path
    ablation_tex: Path
    budget_figure: Path


def magerag_budget_rows(runs: Iterable[RunRecord]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        if run.benchmark != "mmlongbench" or run.baseline != "magerag":
            continue
        if not run.completed_count:
            continue
        params = run.parameters or {}
        metrics = run.metrics or {}
        required = ("top_k", "controller_mode", "watchdog_iterations", "max_selected_actions_per_iteration", "graph_mode")
        if any(key not in params for key in required):
            continue
        rows.append(
            {
                "run_id": run.run_id,
                "top_k": int(params["top_k"]),
                "controller_mode": str(params["controller_mode"]),
                "watchdog_iterations": int(params["watchdog_iterations"]),
                "max_selected_actions_per_iteration": int(params["max_selected_actions_per_iteration"]),
                "graph_mode": str(params["graph_mode"]),
                "overall_acc": _metric(metrics, "overall_acc"),
                "overall_f1": _metric(metrics, "overall_f1"),
                "single_page_acc": _nested_metric(metrics, "breakdowns", "single_page", "acc"),
                "cross_page_acc": _nested_metric(metrics, "breakdowns", "cross_page", "acc"),
                "unanswerable_acc": _nested_metric(metrics, "breakdowns", "unanswerable", "acc"),
                "completed_count": run.completed_count,
                "sample_count": run.sample_count,
                "failed_count": run.failed_count,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["controller_mode"],
            row["graph_mode"],
            row["top_k"],
            row["watchdog_iterations"],
            row["max_selected_actions_per_iteration"],
        ),
    )


def write_budget_ablation_artifacts(
    runs: Iterable[RunRecord],
    output_dir: Path | str,
    paper_image_dir: Path | str,
    paper_table_dir: Path | str | None = None,
) -> PaperArtifacts:
    output = Path(output_dir)
    image_dir = Path(paper_image_dir)
    table_dir = Path(paper_table_dir) if paper_table_dir is not None else output
    output.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    rows = magerag_budget_rows(runs)
    df = pd.DataFrame(rows, columns=_ALL_COLUMNS)
    topk_df = _topk_table(df)
    budget_df = _budget_table(df)
    ablation_df = _ablation_table(df)

    topk_csv = output / "mmlongbench_magerag_topk.csv"
    budget_csv = output / "mmlongbench_magerag_budget.csv"
    ablation_csv = output / "mmlongbench_magerag_ablation.csv"
    topk_tex = table_dir / "mmlongbench_magerag_topk_table.tex"
    budget_tex = table_dir / "mmlongbench_magerag_budget_table.tex"
    ablation_tex = table_dir / "mmlongbench_magerag_ablation_table.tex"
    budget_figure = image_dir / "mmlongbench_magerag_budget_curves.pdf"

    topk_df.to_csv(topk_csv, index=False)
    budget_df.to_csv(budget_csv, index=False)
    ablation_df.to_csv(ablation_csv, index=False)
    topk_tex.write_text(_latex_topk_table(topk_df), encoding="utf-8")
    budget_tex.write_text(_latex_budget_table(budget_df), encoding="utf-8")
    ablation_tex.write_text(_latex_ablation_table(ablation_df), encoding="utf-8")
    _plot_budget_curves(budget_df, budget_figure)

    return PaperArtifacts(
        topk_csv=topk_csv,
        budget_csv=budget_csv,
        ablation_csv=ablation_csv,
        topk_tex=topk_tex,
        budget_tex=budget_tex,
        ablation_tex=ablation_tex,
        budget_figure=budget_figure,
    )


def _topk_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = (
        (df["controller_mode"] == "full")
        & (df["graph_mode"] == "full_graph")
        & (df["watchdog_iterations"] == 5)
        & (df["max_selected_actions_per_iteration"] == 3)
    )
    columns = [
        "top_k",
        "overall_acc",
        "overall_f1",
        "single_page_acc",
        "cross_page_acc",
        "unanswerable_acc",
        "completed_count",
        "sample_count",
        "failed_count",
        "run_id",
    ]
    return df.loc[mask, columns].sort_values(["top_k"])


def _budget_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = (df["controller_mode"] == "full") & (df["graph_mode"] == "full_graph")
    columns = [
        "top_k",
        "watchdog_iterations",
        "max_selected_actions_per_iteration",
        "overall_acc",
        "overall_f1",
        "single_page_acc",
        "cross_page_acc",
        "unanswerable_acc",
        "completed_count",
        "sample_count",
        "failed_count",
        "run_id",
    ]
    return df.loc[mask, columns].sort_values(
        ["top_k", "watchdog_iterations", "max_selected_actions_per_iteration"]
    )


def _ablation_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = (df["top_k"] == 3) & (df["watchdog_iterations"] == 10) & (df["max_selected_actions_per_iteration"] == 5)
    columns = [
        "controller_mode",
        "graph_mode",
        "overall_acc",
        "overall_f1",
        "single_page_acc",
        "cross_page_acc",
        "unanswerable_acc",
        "completed_count",
        "sample_count",
        "failed_count",
        "run_id",
    ]
    order = {
        "topk_page_only": 0,
        "topk_page_with_node_rendering": 1,
        "graph_neighbor_expansion": 2,
        "dynamic_controller_no_search": 3,
        "full": 4,
    }
    table = df.loc[mask, columns].copy()
    table["_order"] = table["controller_mode"].map(order).fillna(10)
    return table.sort_values(["_order", "graph_mode"]).drop(columns=["_order"])


def _latex_topk_table(df: pd.DataFrame) -> str:
    header = r"""\begin{table}[t]
\centering
\caption{MMLongBench-Doc 上不同初始页面 Top-\(k\) 的性能。除 Top-\(k\) 外，其余预算和图结构设置保持一致；Acc/F1 以百分数表示。}
\label{tab:magerag_topk}
\scriptsize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{cccccc}
\hline
Top-\(k\) & Acc & F1 & Single & Cross & Unans. \\
\hline
"""
    body = "".join(
        f"{row.top_k} & {_pct(row.overall_acc)} & {_pct(row.overall_f1)} & "
        f"{_pct(row.single_page_acc)} & {_pct(row.cross_page_acc)} & {_pct(row.unanswerable_acc)} \\\\\n"
        for row in df.itertuples(index=False)
    )
    footer = r"""\hline
\end{tabular}%
}
\end{table}
"""
    return header + body + footer


def _latex_budget_table(df: pd.DataFrame) -> str:
    header = r"""\begin{table}[t]
\centering
\caption{MMLongBench-Doc 上 MAGE-RAG 的预算敏感性结果。所有运行使用 full controller 和 full evidence graph；Acc/F1 以百分数表示。}
\label{tab:magerag_budget}
\scriptsize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{ccccccccc}
\hline
$k$ & $T$ & $a$ & Acc & F1 & Single & Cross & Unans. & Done \\
\hline
"""
    body = "".join(
        f"{row.top_k} & {row.watchdog_iterations} & {row.max_selected_actions_per_iteration} & "
        f"{_pct(row.overall_acc)} & {_pct(row.overall_f1)} & {_pct(row.single_page_acc)} & "
        f"{_pct(row.cross_page_acc)} & {_pct(row.unanswerable_acc)} & "
        f"{int(row.completed_count)}/{int(row.sample_count)} \\\\\n"
        for row in df.itertuples(index=False)
    )
    footer = r"""\hline
\end{tabular}%
}
\end{table}
"""
    return header + body + footer


def _latex_ablation_table(df: pd.DataFrame) -> str:
    header = r"""\begin{table}[t]
\centering
\caption{MMLongBench-Doc 上 MAGE-RAG 的能力与图结构消融。所有运行固定 $k=3,T=10,a=5$；Acc/F1 以百分数表示。}
\label{tab:magerag_ablation}
\scriptsize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llcccccc}
\hline
Controller & Graph & Acc & F1 & Single & Cross & Unans. & Done \\
\hline
"""
    body = "".join(
        f"{_label(row.controller_mode)} & {_label(row.graph_mode)} & {_pct(row.overall_acc)} & "
        f"{_pct(row.overall_f1)} & {_pct(row.single_page_acc)} & {_pct(row.cross_page_acc)} & "
        f"{_pct(row.unanswerable_acc)} & {int(row.completed_count)}/{int(row.sample_count)} \\\\\n"
        for row in df.itertuples(index=False)
    )
    footer = r"""\hline
\end{tabular}%
}
\end{table}
"""
    return header + body + footer


def _plot_budget_curves(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
    if df.empty:
        for axis, title in zip(axes, ["Initial pages $k$", "Controller budget $T$", "Action budget $a$"], strict=True):
            axis.set_title(title)
            axis.text(0.5, 0.5, "No completed runs", ha="center", va="center", transform=axis.transAxes)
            axis.set_axis_off()
        fig.savefig(path)
        plt.close(fig)
        return
    series = [
        (df[(df["watchdog_iterations"] == 5) & (df["max_selected_actions_per_iteration"] == 3)], "top_k", "Initial pages $k$"),
        (df[(df["top_k"] == 3) & (df["max_selected_actions_per_iteration"] == 3)], "watchdog_iterations", "Controller budget $T$"),
        (df[(df["top_k"] == 3) & (df["watchdog_iterations"] == 5)], "max_selected_actions_per_iteration", "Action budget $a$"),
    ]
    for axis, (subset, x_key, title) in zip(axes, series, strict=True):
        subset = subset.sort_values(x_key)
        axis.plot(subset[x_key], subset["overall_acc"] * 100, marker="o", label="Overall Acc")
        axis.plot(subset[x_key], subset["cross_page_acc"] * 100, marker="s", label="Cross-page Acc")
        axis.set_title(title)
        axis.set_xlabel(x_key.replace("_", " "))
        axis.set_ylabel("Score (%)")
        axis.grid(True, linewidth=0.4, alpha=0.4)
    axes[0].legend(loc="best", fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def _metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if isinstance(value, int | float) else None


def _nested_metric(metrics: dict[str, Any], *keys: str) -> float | None:
    node: Any = metrics
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return float(node) if isinstance(node, int | float) else None


def _pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value) * 100:.2f}"


def _label(value: Any) -> str:
    return str(value).replace("_", " ").title()


_ALL_COLUMNS = [
    "run_id",
    "top_k",
    "controller_mode",
    "watchdog_iterations",
    "max_selected_actions_per_iteration",
    "graph_mode",
    "overall_acc",
    "overall_f1",
    "single_page_acc",
    "cross_page_acc",
    "unanswerable_acc",
    "completed_count",
    "sample_count",
    "failed_count",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper artifacts for MAGE-RAG budget and ablation analysis.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_cache/paper"))
    parser.add_argument("--paper-image-dir", type=Path, default=Path("../paper/images"))
    parser.add_argument("--paper-table-dir", type=Path, default=Path("../paper/tables"))
    args = parser.parse_args()

    artifacts = write_budget_ablation_artifacts(
        scan_runs(args.results_root),
        args.output_dir,
        args.paper_image_dir,
        args.paper_table_dir,
    )
    for path in artifacts.__dict__.values():
        print(path)


if __name__ == "__main__":
    main()
