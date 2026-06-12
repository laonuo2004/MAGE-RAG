from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from analysis.paper_registry import official_paper_runs
from analysis.results_loader import RunRecord, read_jsonl_cached
from analysis.results_metrics import pairwise_comparison, retrieval_diagnostics


@dataclass(frozen=True)
class PaperDiagnosticArtifacts:
    retrieval_csv: Path
    pairwise_csv: Path


def write_paper_diagnostic_artifacts(
    runs: Iterable[RunRecord],
    output_dir: Path | str,
) -> PaperDiagnosticArtifacts:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    selected = official_paper_runs(runs)
    retrieval_rows = _retrieval_rows(selected, output)
    pairwise_rows = _pairwise_rows(selected, output)

    retrieval_csv = output / "paper_retrieval_diagnostics.csv"
    pairwise_csv = output / "paper_magerag_pairwise_summary.csv"
    pd.DataFrame(retrieval_rows).to_csv(retrieval_csv, index=False)
    pd.DataFrame(pairwise_rows).to_csv(pairwise_csv, index=False)
    return PaperDiagnosticArtifacts(retrieval_csv=retrieval_csv, pairwise_csv=pairwise_csv)


def _retrieval_rows(
    selected: dict[tuple[str, str], RunRecord],
    output: Path,
) -> list[dict]:
    rows: list[dict] = []
    for (benchmark, baseline), run in sorted(selected.items()):
        if run.jsonl_path is None:
            continue
        records = read_jsonl_cached(run.jsonl_path, cache_root=output / "result_cache", cache_namespace=run.run_id)
        for row in retrieval_diagnostics(records):
            rows.append(
                {
                    "benchmark": benchmark,
                    "baseline": baseline,
                    "run_id": run.run_id,
                    **row,
                }
            )
    return rows


def _pairwise_rows(
    selected: dict[tuple[str, str], RunRecord],
    output: Path,
) -> list[dict]:
    rows: list[dict] = []
    benchmarks = sorted({benchmark for benchmark, _baseline in selected})
    for benchmark in benchmarks:
        magerag = selected.get((benchmark, "magerag"))
        if magerag is None or magerag.jsonl_path is None:
            continue
        magerag_records = read_jsonl_cached(
            magerag.jsonl_path,
            cache_root=output / "result_cache",
            cache_namespace=magerag.run_id,
        )
        for (other_benchmark, baseline), run in sorted(selected.items()):
            if other_benchmark != benchmark or baseline == "magerag" or run.jsonl_path is None:
                continue
            comparison = pairwise_comparison(magerag_records, read_jsonl_cached(
                run.jsonl_path,
                cache_root=output / "result_cache",
                cache_namespace=run.run_id,
            ))
            summary = comparison["summary"]
            rows.append(
                {
                    "benchmark": benchmark,
                    "baseline": baseline,
                    "magerag_run_id": magerag.run_id,
                    "baseline_run_id": run.run_id,
                    "a_wins": summary["a_wins"],
                    "b_wins": summary["b_wins"],
                    "ties": summary["ties"],
                    "aligned": summary["aligned"],
                }
            )
    return rows
