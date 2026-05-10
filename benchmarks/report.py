import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from benchmarks.results import RESULTS_ROOT, iter_result_manifests, utc_now, write_json


PRIMARY_METRICS = ("overall_acc", "avg_acc", "overall_f1", "f1", "rectified_avg_acc")


def _metric_value(metrics: Dict[str, Any] | None) -> tuple[str | None, float | None]:
    if not metrics:
        return None, None
    for key in PRIMARY_METRICS:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return key, float(value)
    return None, None


def _run_record(manifest: Dict[str, Any], metrics: Dict[str, Any] | None) -> Dict[str, Any]:
    metric_name, metric_value = _metric_value(metrics)
    results_file = Path(manifest.get("results_file", ""))
    sample_count = metrics.get("sample_count") if metrics else None
    completed_count = metrics.get("completed_count") if metrics else None
    failed_count = metrics.get("failed_count") if metrics else None
    return {
        "benchmark": manifest.get("benchmark"),
        "baseline": manifest.get("baseline"),
        "qa_model_name": manifest.get("qa_model_name"),
        "parameters": manifest.get("parameters", {}),
        "results_file": str(results_file),
        "metrics_file": manifest.get("metrics_file"),
        "metrics": metrics,
        "status": "ok" if metrics is not None else "missing_metrics",
        "primary_metric": metric_name,
        "primary_metric_value": metric_value,
        "sample_count": sample_count,
        "completed_count": completed_count,
        "failed_count": failed_count,
        "updated_at": manifest.get("generated_at"),
    }


def _best_by_metric(runs: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for run in runs:
        metric = run.get("primary_metric")
        value = run.get("primary_metric_value")
        if metric is None or value is None:
            continue
        current = best.get(metric)
        if current is None or value > current.get("primary_metric_value", float("-inf")):
            best[metric] = run
    return best


def _build_report(scope: Dict[str, Any], runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    benchmarks = sorted({run.get("benchmark") for run in runs if run.get("benchmark")})
    baselines = sorted({run.get("baseline") for run in runs if run.get("baseline")})
    return {
        "generated_at": utc_now(),
        "scope": scope,
        "benchmarks": benchmarks,
        "baselines": baselines,
        "runs": runs,
        "best_by_metric": _best_by_metric(runs),
    }


def generate_reports(results_root: str | Path = RESULTS_ROOT) -> Dict[str, Any]:
    root = Path(results_root)
    all_runs = [
        _run_record(manifest, metrics)
        for _, manifest, metrics in iter_result_manifests(root)
    ]
    root_report = _build_report({"level": "root", "path": str(root)}, all_runs)
    write_json(root / "report.json", root_report)

    for benchmark in sorted({run["benchmark"] for run in all_runs if run.get("benchmark")}):
        benchmark_runs = [run for run in all_runs if run.get("benchmark") == benchmark]
        write_json(
            root / benchmark / "report.json",
            _build_report({"level": "benchmark", "benchmark": benchmark, "path": str(root / benchmark)}, benchmark_runs),
        )
        for baseline in sorted({run["baseline"] for run in benchmark_runs if run.get("baseline")}):
            baseline_runs = [run for run in benchmark_runs if run.get("baseline") == baseline]
            write_json(
                root / benchmark / baseline / "report.json",
                _build_report(
                    {
                        "level": "baseline",
                        "benchmark": benchmark,
                        "baseline": baseline,
                        "path": str(root / benchmark / baseline),
                    },
                    baseline_runs,
                ),
            )
    return root_report


if __name__ == "__main__":
    print(json.dumps(generate_reports(), ensure_ascii=False, indent=2))
