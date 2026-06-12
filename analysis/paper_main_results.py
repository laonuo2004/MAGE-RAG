from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from analysis.paper_registry import official_paper_runs
from analysis.results_loader import RunRecord, scan_runs


METHODS = [
    ("image", "Direct MLLM", "Direct MLLM"),
    ("bm25", "BM25", "Text RAG"),
    ("colbertv2", "ColBERTv2", "Text RAG"),
    ("m3docrag", "M3DocRAG", "Page-level Visual RAG"),
    ("evisrag", "EVisRAG", "Page-level Visual RAG"),
    ("g2_reader", "G2Reader", "Graph/Agentic RAG"),
    ("magerag", "MAGE-RAG", "Graph/Agentic RAG"),
]

MLDOCRAG_ROW = {
    "method": "MLDocRAG",
    "type": "Graph/Agentic RAG",
    "source": "Published",
    "reader": "--",
    "longdocurl_text": 0.651,
    "longdocurl_layout": 0.394,
    "longdocurl_figure": 0.483,
    "longdocurl_table": 0.411,
    "longdocurl_single_page": 0.669,
    "longdocurl_multi_page": 0.563,
    "longdocurl_cross_element": 0.234,
    "longdocurl_overall": 0.508,
    "longdocurl_done": "--",
    "mmlongbench_text": 0.472,
    "mmlongbench_layout": 0.378,
    "mmlongbench_chart": 0.427,
    "mmlongbench_table": 0.413,
    "mmlongbench_figure": 0.319,
    "mmlongbench_single_page": 0.526,
    "mmlongbench_cross_page": 0.264,
    "mmlongbench_unanswerable": 0.715,
    "mmlongbench_acc": 0.479,
    "mmlongbench_f1": None,
    "mmlongbench_done": "--",
}

METRIC_COLUMNS = [
    "longdocurl_text",
    "longdocurl_layout",
    "longdocurl_figure",
    "longdocurl_table",
    "longdocurl_single_page",
    "longdocurl_multi_page",
    "longdocurl_cross_element",
    "longdocurl_overall",
    "mmlongbench_text",
    "mmlongbench_layout",
    "mmlongbench_chart",
    "mmlongbench_table",
    "mmlongbench_figure",
    "mmlongbench_single_page",
    "mmlongbench_cross_page",
    "mmlongbench_unanswerable",
    "mmlongbench_acc",
    "mmlongbench_f1",
]


@dataclass(frozen=True)
class MainResultsArtifacts:
    csv_path: Path
    tex_path: Path


def main_result_rows(runs: Iterable[RunRecord]) -> list[dict[str, Any]]:
    best = _best_runs(runs)
    rows = []
    for baseline, method, method_type in METHODS:
        longdoc = best.get(("longdocurl", baseline))
        mmlong = best.get(("mmlongbench", baseline))
        row = {
            "method": method,
            "type": method_type,
            "source": "Reproduced",
            "reader": "Qwen3-VL-8B",
        }
        row.update(_longdocurl_columns(longdoc))
        row.update(_mmlongbench_columns(mmlong))
        rows.append(row)
    rows.append(dict(MLDOCRAG_ROW))
    return rows


def write_main_results_artifacts(
    runs: Iterable[RunRecord],
    output_dir: Path | str,
    paper_table_dir: Path | str,
) -> MainResultsArtifacts:
    output = Path(output_dir)
    table_dir = Path(paper_table_dir)
    output.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    rows = main_result_rows(runs)
    df = pd.DataFrame(rows)
    csv_path = output / "main_results.csv"
    tex_path = table_dir / "main_results_table.tex"
    df.to_csv(csv_path, index=False)
    tex_path.write_text(_latex_main_table(rows), encoding="utf-8")
    return MainResultsArtifacts(csv_path=csv_path, tex_path=tex_path)


def _best_runs(runs: Iterable[RunRecord]) -> dict[tuple[str, str], RunRecord]:
    allowed = {method[0] for method in METHODS}
    return {
        key: run
        for key, run in official_paper_runs(runs).items()
        if run.baseline in allowed
    }


def _official_score(run: RunRecord) -> float | None:
    metrics = run.metrics or {}
    for key in ("overall_acc", "avg_acc", "acc"):
        value = metrics.get(key)
        if isinstance(value, int | float):
            return float(value)
    return None


def _longdocurl_columns(run: RunRecord | None) -> dict[str, Any]:
    if run is None:
        return {
            "longdocurl_overall": None,
            "longdocurl_text": None,
            "longdocurl_layout": None,
            "longdocurl_figure": None,
            "longdocurl_table": None,
            "longdocurl_understanding": None,
            "longdocurl_reasoning": None,
            "longdocurl_locating": None,
            "longdocurl_single_page": None,
            "longdocurl_multi_page": None,
            "longdocurl_cross_element": None,
            "longdocurl_done": None,
        }
    metrics = run.metrics or {}
    fine = metrics.get("fine_grained_metrics") if isinstance(metrics.get("fine_grained_metrics"), dict) else {}
    main_task = fine.get("Main_Task") if isinstance(fine.get("Main_Task"), dict) else {}
    element_type = fine.get("Element_Type") if isinstance(fine.get("Element_Type"), dict) else {}
    evidence_pages = fine.get("Evidence_Pages") if isinstance(fine.get("Evidence_Pages"), dict) else {}
    num_element_types = fine.get("Num_of_Element_Types") if isinstance(fine.get("Num_of_Element_Types"), dict) else {}
    return {
        "longdocurl_overall": _metric(metrics, "overall_acc", "avg_acc"),
        "longdocurl_text": _nested_metric(element_type, "Text"),
        "longdocurl_layout": _nested_metric(element_type, "Layout"),
        "longdocurl_figure": _nested_metric(element_type, "Figure"),
        "longdocurl_table": _nested_metric(element_type, "Table"),
        "longdocurl_understanding": _nested_metric(main_task, "Understanding"),
        "longdocurl_reasoning": _nested_metric(main_task, "Reasoning"),
        "longdocurl_locating": _nested_metric(main_task, "Locating"),
        "longdocurl_single_page": _nested_metric(evidence_pages, "Single_Page"),
        "longdocurl_multi_page": _nested_metric(evidence_pages, "Multi_Page"),
        "longdocurl_cross_element": _nested_metric(num_element_types, "Cross_Element"),
        "longdocurl_done": _done(run),
    }


def _mmlongbench_columns(run: RunRecord | None) -> dict[str, Any]:
    if run is None:
        return {
            "mmlongbench_acc": None,
            "mmlongbench_f1": None,
            "mmlongbench_text": None,
            "mmlongbench_layout": None,
            "mmlongbench_chart": None,
            "mmlongbench_table": None,
            "mmlongbench_figure": None,
            "mmlongbench_single_page": None,
            "mmlongbench_cross_page": None,
            "mmlongbench_unanswerable": None,
            "mmlongbench_done": None,
        }
    metrics = run.metrics or {}
    source_breakdowns = metrics.get("evidence_source_breakdowns")
    if not isinstance(source_breakdowns, dict):
        source_breakdowns = {}
    page_breakdowns = metrics.get("breakdowns")
    if not isinstance(page_breakdowns, dict):
        page_breakdowns = {}
    return {
        "mmlongbench_acc": _metric(metrics, "overall_acc"),
        "mmlongbench_f1": _metric(metrics, "overall_f1"),
        "mmlongbench_text": _nested_metric(source_breakdowns, "Pure-text (Plain-text)", "acc"),
        "mmlongbench_layout": _nested_metric(source_breakdowns, "Generalized-text (Layout)", "acc"),
        "mmlongbench_chart": _nested_metric(source_breakdowns, "Chart", "acc"),
        "mmlongbench_table": _nested_metric(source_breakdowns, "Table", "acc"),
        "mmlongbench_figure": _nested_metric(source_breakdowns, "Figure", "acc"),
        "mmlongbench_single_page": _nested_metric(page_breakdowns, "single_page", "acc"),
        "mmlongbench_cross_page": _nested_metric(page_breakdowns, "cross_page", "acc"),
        "mmlongbench_unanswerable": _nested_metric(page_breakdowns, "unanswerable", "acc"),
        "mmlongbench_done": _done(run),
    }


def _latex_main_table(rows: list[dict[str, Any]]) -> str:
    header = r"""\begin{table*}[t]
\centering
\caption{LongDocURL 与 MMLongBench-Doc 上的主实验结果。除 MLDocRAG 外，所有结果均由当前仓库 \texttt{results/} 中已完成样本的 metrics 生成；数值为百分数。}
\label{tab:main_results}
\setlength{\tabcolsep}{2.7pt}
\renewcommand{\arraystretch}{1.05}
\scriptsize
\resizebox{\textwidth}{!}{%
\begin{tabular}{lll|cccccccc|cccccccccc}
\hline\hline
\multicolumn{3}{c|}{Method} &
\multicolumn{8}{c|}{LongDocURL} &
\multicolumn{10}{c}{MMLongBench-Doc} \\
\cline{1-21}
Method & Type & Source &
\multicolumn{4}{c|}{Evidence Source} &
\multicolumn{3}{c|}{Evidence Page} & Overall &
\multicolumn{5}{c|}{Evidence Source} &
\multicolumn{3}{c|}{Evidence Page} & Acc & F1 \\
\cline{4-10}\cline{12-19}
 & & & TXT & LAY & FIG & TAB & SP & MP & CE & Acc
 & TXT & LAY & CHA & TAB & FIG & SIN & MUL & UNA & & \\
\hline
"""
    highlights = _rank_highlights(rows)
    body_parts = []
    previous_type = None
    for row in rows:
        if previous_type is not None and row["type"] != previous_type:
            body_parts.append("\\hline\n")
        previous_type = row["type"]
        body_parts.append(
            f"{row['method']} & {row['type']} & {row['source']} & "
            f"{_cell(row, 'longdocurl_text', highlights)} & {_cell(row, 'longdocurl_layout', highlights)} & "
            f"{_cell(row, 'longdocurl_figure', highlights)} & {_cell(row, 'longdocurl_table', highlights)} & "
            f"{_cell(row, 'longdocurl_single_page', highlights)} & {_cell(row, 'longdocurl_multi_page', highlights)} & "
            f"{_cell(row, 'longdocurl_cross_element', highlights)} & {_cell(row, 'longdocurl_overall', highlights)} & "
            f"{_cell(row, 'mmlongbench_text', highlights)} & {_cell(row, 'mmlongbench_layout', highlights)} & "
            f"{_cell(row, 'mmlongbench_chart', highlights)} & {_cell(row, 'mmlongbench_table', highlights)} & "
            f"{_cell(row, 'mmlongbench_figure', highlights)} & {_cell(row, 'mmlongbench_single_page', highlights)} & "
            f"{_cell(row, 'mmlongbench_cross_page', highlights)} & {_cell(row, 'mmlongbench_unanswerable', highlights)} & "
            f"{_cell(row, 'mmlongbench_acc', highlights)} & {_cell(row, 'mmlongbench_f1', highlights)} \\\\\n"
        )
    body = "".join(body_parts)
    done_rows = [
        f"{row['method']}: LongDocURL {row['longdocurl_done'] or '--'}, MMLongBench {row['mmlongbench_done'] or '--'}"
        for row in rows
    ]
    footer_prefix = r"""\hline\hline
\end{tabular}%
}
\vspace{1mm}
\begin{minipage}{0.98\textwidth}
\scriptsize \textit{Notes.} TXT, LAY, CHA, FIG, and TAB denote text, layout, chart, figure, and table evidence sources. SP/SIN, MP/MUL, CE, and UNA denote single-page, multi-page, cross-element, and unanswerable subsets. Done counts: """
    footer_suffix = r""".
\end{minipage}
\end{table*}
"""
    footer = footer_prefix + "; ".join(done_rows) + footer_suffix
    return header + body + footer


def _metric(metrics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, int | float):
            return float(value)
    return None


def _nested_metric(node: dict[str, Any], *keys: str) -> float | None:
    value: Any = node
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return float(value) if isinstance(value, int | float) else None


def _done(run: RunRecord) -> str:
    if run.completed_count is None or run.sample_count is None:
        return "--"
    return f"{run.completed_count}/{run.sample_count}"


def _pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value) * 100:.2f}"


def _cell(row: dict[str, Any], key: str, highlights: dict[str, dict[str, float]]) -> str:
    value = row.get(key)
    text = _pct(value)
    if text == "--":
        return text
    highlight = highlights.get(key, {})
    numeric = float(value)
    if numeric == highlight.get("best"):
        return rf"\textbf{{{text}}}"
    if numeric == highlight.get("second"):
        return rf"\underline{{{text}}}"
    return text


def _rank_highlights(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    highlights: dict[str, dict[str, float]] = {}
    for key in METRIC_COLUMNS:
        values = sorted(
            {
                float(row[key])
                for row in rows
                if key in row and row[key] is not None and not pd.isna(row[key])
            },
            reverse=True,
        )
        if values:
            highlights[key] = {"best": values[0]}
            if len(values) > 1:
                highlights[key]["second"] = values[1]
    return highlights


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper main-result table from current results metrics.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_cache/paper"))
    parser.add_argument("--paper-table-dir", type=Path, default=Path("../paper/tables"))
    args = parser.parse_args()

    artifacts = write_main_results_artifacts(scan_runs(args.results_root), args.output_dir, args.paper_table_dir)
    for path in artifacts.__dict__.values():
        print(path)


if __name__ == "__main__":
    main()
