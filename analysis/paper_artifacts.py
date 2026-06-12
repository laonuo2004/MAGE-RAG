from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analysis.paper_breakdown import write_breakdown_artifacts
from analysis.paper_budget_ablation import write_budget_ablation_artifacts
from analysis.paper_case_study import write_case_study_artifacts
from analysis.paper_diagnostics import write_paper_diagnostic_artifacts
from analysis.paper_efficiency import write_efficiency_artifacts
from analysis.paper_main_results import main_result_rows, write_main_results_artifacts
from analysis.paper_registry import paper_run_diagnostics, selected_run_manifest
from analysis.paper_trace_statistics import write_trace_statistics_artifacts
from analysis.results_loader import scan_runs


DEFAULT_TRACE_JSONL = Path(
    "results/mmlongbench/magerag/"
    "res_top_k_3_mode_full_watchdog_iterations_5_max_selected_actions_per_iteration_3_mode_full_graph_Qwen3_VL_8B_Instruct.jsonl"
)
DEFAULT_CASE_JSONL = Path(
    "results/mmlongbench/magerag/"
    "res_top_k_3_mode_full_watchdog_iterations_5_max_selected_actions_per_iteration_8_mode_full_graph_Qwen3_VL_8B_Instruct.jsonl"
)


def write_all_paper_artifacts(
    results_root: Path | str = Path("results"),
    output_dir: Path | str = Path("analysis_cache/paper"),
    paper_image_dir: Path | str = Path("../paper/images"),
    paper_table_dir: Path | str = Path("../paper/tables"),
    trace_jsonl_path: Path | str | None = DEFAULT_TRACE_JSONL,
    case_jsonl_path: Path | str | None = DEFAULT_CASE_JSONL,
    efficiency_jsonl_path: Path | str | None = DEFAULT_TRACE_JSONL,
    strict: bool = False,
    include_partial: bool = False,
) -> dict[str, Any]:
    results_root = Path(results_root)
    output_dir = Path(output_dir)
    paper_image_dir = Path(paper_image_dir)
    paper_table_dir = Path(paper_table_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_image_dir.mkdir(parents=True, exist_ok=True)
    paper_table_dir.mkdir(parents=True, exist_ok=True)

    runs = scan_runs(results_root)
    run_diagnostics = paper_run_diagnostics(runs)
    selected_runs = selected_run_manifest(runs)
    warnings = _manifest_warnings(run_diagnostics, include_partial=include_partial)
    if strict:
        _raise_for_strict_failures(run_diagnostics, selected_runs)
    main_rows = main_result_rows(runs)

    artifacts: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []

    _extend_artifacts(artifacts, "main_results", write_main_results_artifacts(runs, output_dir, paper_table_dir))
    _extend_artifacts(artifacts, "breakdown", write_breakdown_artifacts(main_rows, output_dir, paper_image_dir, paper_table_dir))
    _extend_artifacts(artifacts, "budget_ablation", write_budget_ablation_artifacts(runs, output_dir, paper_image_dir, paper_table_dir))
    _extend_artifacts(artifacts, "diagnostics", write_paper_diagnostic_artifacts(runs, output_dir))

    trace_path = Path(trace_jsonl_path) if trace_jsonl_path is not None else None
    if trace_path and trace_path.exists():
        _extend_artifacts(
            artifacts,
            "trace_statistics",
            write_trace_statistics_artifacts(trace_path, output_dir, paper_image_dir, paper_table_dir),
        )
    else:
        skipped.append({"name": "trace_statistics", "reason": "missing input jsonl"})

    efficiency_path = Path(efficiency_jsonl_path) if efficiency_jsonl_path is not None else None
    if efficiency_path and efficiency_path.exists():
        _extend_artifacts(
            artifacts,
            "efficiency",
            write_efficiency_artifacts(efficiency_path, output_dir, paper_image_dir, paper_table_dir),
        )
    else:
        skipped.append({"name": "efficiency", "reason": "missing input jsonl"})

    case_path = Path(case_jsonl_path) if case_jsonl_path is not None else None
    if case_path and case_path.exists():
        _extend_artifacts(
            artifacts,
            "case_study",
            write_case_study_artifacts(case_path, output_dir, paper_image_dir, paper_table_dir),
        )
    else:
        skipped.append({"name": "case_study", "reason": "missing input jsonl"})

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "results_root": str(results_root),
            "trace_jsonl_path": str(trace_path) if trace_path else None,
            "case_jsonl_path": str(case_path) if case_path else None,
            "efficiency_jsonl_path": str(efficiency_path) if efficiency_path else None,
        },
        "artifact_count": len(artifacts),
        "run_diagnostics": run_diagnostics,
        "selected_runs": selected_runs,
        "warnings": warnings,
        "artifacts": artifacts,
        "skipped": skipped,
    }
    manifest_path = output_dir / "paper_artifacts_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _extend_artifacts(artifacts: list[dict[str, str]], prefix: str, artifact_group: Any) -> None:
    if is_dataclass(artifact_group):
        values = asdict(artifact_group)
    elif isinstance(artifact_group, dict):
        values = artifact_group
    else:
        values = getattr(artifact_group, "__dict__", {})
    for key, value in values.items():
        path = Path(value)
        artifacts.append({"name": f"{prefix}.{key}", "path": str(path)})


def _manifest_warnings(run_diagnostics: list[dict[str, Any]], include_partial: bool) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    for item in run_diagnostics:
        status = str(item.get("status") or "")
        if status == "partial" and include_partial:
            continue
        if status in {"partial", "empty", "missing_metrics", "invalid_schema"}:
            warnings.append(
                {
                    "run_id": str(item.get("run_id")),
                    "status": status,
                    "message": "; ".join(map(str, item.get("warnings") or [])),
                }
            )
    return warnings


def _raise_for_strict_failures(run_diagnostics: list[dict[str, Any]], selected_runs: list[dict[str, Any]]) -> None:
    selected_keys = {(row.get("benchmark"), row.get("baseline")) for row in selected_runs}
    required_keys = {
        (item.get("benchmark"), item.get("baseline"))
        for item in run_diagnostics
        if not item.get("paper_excluded")
    }
    missing = sorted(required_keys - selected_keys)
    if missing:
        formatted = ", ".join(f"{benchmark}/{baseline}" for benchmark, baseline in missing)
        raise RuntimeError(f"Strict paper artifact generation missing selected runs for: {formatted}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate all paper-facing experiment analysis artifacts.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_cache/paper"))
    parser.add_argument("--paper-image-dir", type=Path, default=Path("../paper/images"))
    parser.add_argument("--paper-table-dir", type=Path, default=Path("../paper/tables"))
    parser.add_argument("--trace-jsonl-path", type=Path, default=DEFAULT_TRACE_JSONL)
    parser.add_argument("--case-jsonl-path", type=Path, default=DEFAULT_CASE_JSONL)
    parser.add_argument("--efficiency-jsonl-path", type=Path, default=DEFAULT_TRACE_JSONL)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--include-partial", action="store_true")
    args = parser.parse_args()

    manifest = write_all_paper_artifacts(
        results_root=args.results_root,
        output_dir=args.output_dir,
        paper_image_dir=args.paper_image_dir,
        paper_table_dir=args.paper_table_dir,
        trace_jsonl_path=args.trace_jsonl_path,
        case_jsonl_path=args.case_jsonl_path,
        efficiency_jsonl_path=args.efficiency_jsonl_path,
        strict=args.strict,
        include_partial=args.include_partial,
    )
    print(Path(args.output_dir) / "paper_artifacts_manifest.json")
    for artifact in manifest["artifacts"]:
        print(artifact["path"])
    for skipped in manifest["skipped"]:
        print(f"SKIPPED {skipped['name']}: {skipped['reason']}")


if __name__ == "__main__":
    main()
