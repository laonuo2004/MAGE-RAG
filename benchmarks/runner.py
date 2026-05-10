import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List

from tqdm import tqdm

from benchmarks.adapters import BenchmarkAdapter
from benchmarks.report import generate_reports
from benchmarks.results import (
    append_jsonl,
    build_manifest,
    build_results_file,
    read_jsonl,
    sidecar_paths,
    write_json,
)
from utils.config_utils import get_config_value

logger = logging.getLogger(__name__)

MAX_RETRY_ROUNDS = 3


def successful_results(adapter: BenchmarkAdapter, output_path: str | Path) -> List[Dict[str, Any]]:
    return [sample for sample in read_jsonl(output_path) if adapter.is_successful_result(sample)]


def compact_results_file(adapter: BenchmarkAdapter, output_path: str | Path) -> None:
    output_path = Path(output_path)
    if not output_path.exists():
        return
    raw = read_jsonl(output_path)
    unique = {}
    for sample in raw:
        if adapter.is_successful_result(sample):
            unique[adapter.sample_key(sample)] = sample
    if len(unique) == len(raw):
        return
    with output_path.open("w", encoding="utf-8") as f:
        for sample in unique.values():
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def merge_existing_samples(adapter: BenchmarkAdapter, samples: List[Dict[str, Any]], output_path: str | Path) -> List[Dict[str, Any]]:
    existing_by_key = {adapter.sample_key(sample): sample for sample in successful_results(adapter, output_path)}
    merged = []
    for sample in samples:
        existing = existing_by_key.get(adapter.sample_key(sample))
        if existing is None:
            merged.append(sample)
            continue
        merged_sample = dict(sample)
        merged_sample.update(existing)
        merged.append(merged_sample)
    return merged


def run_pending(adapter: BenchmarkAdapter, cfg, samples: List[Dict[str, Any]], output_path: Path) -> None:
    process_mode = get_config_value(cfg, "benchmarks.process_mode", "serial")
    workers = int(get_config_value(cfg, "benchmarks.workers", 1))
    pending_indices = [idx for idx, sample in enumerate(samples) if not adapter.is_successful_result(sample)]
    logger.info("Progress: %s/%s completed", len(samples) - len(pending_indices), len(samples))
    logger.info("Evaluation Process Mode: %s", process_mode)
    if process_mode == "parallel":
        logger.info("Number Of Worker Threads: %s", workers)
    if not pending_indices:
        logger.info("No Data To Process.")
        return

    def run_index(idx: int) -> Dict[str, Any] | None:
        started = perf_counter()
        try:
            result = adapter.process_sample(samples[idx], cfg)
        except Exception:
            logger.exception(
                "%s sample failed. idx=%s sample_key=%s",
                adapter.name,
                idx,
                adapter.sample_key(samples[idx]),
            )
            return None
        if result is not None:
            result.setdefault("timing_total_seconds", round(perf_counter() - started, 3))
        return result

    def handle_result(idx: int, result: Dict[str, Any] | None) -> None:
        if result is None or not adapter.is_successful_result(result):
            logger.warning(
                "Skipping failed %s sample without writing JSONL. idx=%s sample_key=%s",
                adapter.name,
                idx,
                adapter.sample_key(samples[idx]),
            )
            return
        samples[idx] = result
        append_jsonl(result, output_path)

    if process_mode == "serial":
        for idx in pending_indices:
            handle_result(idx, run_index(idx))
        return

    if process_mode == "parallel":
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            future_to_index = {executor.submit(run_index, idx): idx for idx in pending_indices}
            for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Processing", mininterval=1):
                idx = future_to_index[future]
                handle_result(idx, future.result())
        return

    raise ValueError(f"Unsupported process_mode: {process_mode}")


def run_benchmark_with_adapter(cfg, adapter: BenchmarkAdapter) -> Dict[str, Any]:
    configured_results_file = get_config_value(cfg, "benchmarks.results_file")
    output_path = Path(configured_results_file) if configured_results_file else build_results_file(cfg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Output Datapath: %s", output_path)
    compact_results_file(adapter, output_path)

    samples: List[Dict[str, Any]] = []
    for round_id in range(1, MAX_RETRY_ROUNDS + 1):
        samples = merge_existing_samples(adapter, adapter.load_samples(cfg), output_path)
        completed = sum(1 for sample in samples if adapter.is_successful_result(sample))
        if completed == len(samples):
            break
        logger.info(
            "%s retry round %s: %s/%s successful samples found in JSONL",
            adapter.name,
            round_id,
            completed,
            len(samples),
        )
        run_pending(adapter, cfg, samples, output_path)

    samples = merge_existing_samples(adapter, adapter.load_samples(cfg), output_path)
    completed = sum(1 for sample in samples if adapter.is_successful_result(sample))
    failed = len(samples) - completed
    if failed:
        logger.warning("%s stopped with %s/%s successful samples.", adapter.name, completed, len(samples))

    metrics = adapter.build_metrics(samples, output_path)
    metrics.setdefault("sample_count", len(samples))
    metrics.setdefault("completed_count", completed)
    metrics.setdefault("failed_count", failed)

    metrics_path, manifest_path = sidecar_paths(output_path)
    write_json(metrics_path, metrics)
    write_json(manifest_path, build_manifest(cfg, output_path, metrics_path))
    generate_reports(output_path.parents[2])

    return {
        "results_file": str(output_path),
        "metrics_file": str(metrics_path),
        "manifest_file": str(manifest_path),
        "metrics": metrics,
        "samples": samples,
    }
