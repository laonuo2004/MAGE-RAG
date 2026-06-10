import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from benchmarks.adapters import BenchmarkAdapter, _finalize_result_fields
from benchmarks.utils.results_utils import (
    append_jsonl,
    build_results_file,
    read_jsonl,
    write_json,
)
from baselines.wrapper import build_context_builder
from utils.config_utils import get_config_value
from utils.llm_utils import build_openai_client

logger = logging.getLogger(__name__)

MAX_RETRY_ROUNDS = 3


def overwrite_enabled(cfg) -> bool:
    return bool(get_config_value(cfg, "overwrite", False))


def clear_existing_results(output_path: str | Path) -> None:
    output_path = Path(output_path)
    for path in (output_path, output_path.with_suffix(".metrics.json")):
        if path.exists():
            path.unlink()
            logger.info("Removed existing result artifact for overwrite: %s", path)


def successful_results(adapter: BenchmarkAdapter, output_path: str | Path) -> List[Dict[str, Any]]:
    return [sample for sample in read_jsonl(output_path) if adapter.is_successful_result(sample)]


def compact_results_file(adapter: BenchmarkAdapter, output_path: str | Path) -> None:
    output_path = Path(output_path)
    if not output_path.exists():
        return
    original_text = output_path.read_text(encoding="utf-8")
    raw = read_jsonl(output_path)
    unique = {}
    for sample in raw:
        if adapter.is_successful_result(sample):
            unique[adapter.sample_key(sample)] = _finalize_result_fields(dict(sample))
    compacted_text = "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in unique.values())
    if compacted_text == original_text:
        return
    output_path.write_text(compacted_text, encoding="utf-8")


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


def finalize_metrics(adapter: BenchmarkAdapter, cfg, output_path: str | Path) -> Dict[str, Any]:
    output_path = Path(output_path)
    samples = merge_existing_samples(adapter, adapter.load_samples(cfg), output_path)
    completed = sum(1 for sample in samples if adapter.is_successful_result(sample))
    failed = len(samples) - completed

    metrics = adapter.build_metrics(samples, output_path)
    metrics.setdefault("sample_count", len(samples))
    metrics.setdefault("completed_count", completed)
    metrics.setdefault("failed_count", failed)

    metrics_path = output_path.with_suffix(".metrics.json")
    write_json(metrics_path, metrics)
    logger.info("Metrics saved to %s", metrics_path)
    return {
        "metrics_file": str(metrics_path),
        "metrics": metrics,
        "samples": samples,
    }


def run_pending(
    adapter: BenchmarkAdapter,
    cfg,
    samples: List[Dict[str, Any]],
    output_path: Path,
    context_builder,
    client,
) -> None:
    process_mode = get_config_value(cfg, "benchmarks.process_mode", "serial")
    workers = int(get_config_value(cfg, "benchmarks.workers", 1))
    pending_indices = [idx for idx, sample in enumerate(samples) if not adapter.is_successful_result(sample)]
    logger.info("Evaluation Process Mode: %s", process_mode)
    if process_mode == "parallel":
        logger.info("Number Of Worker Threads: %s", workers)
    if not pending_indices:
        logger.info("No Data To Process.")
        return

    def run_index(idx: int) -> Dict[str, Any] | None:
        try:
            result = adapter.process_sample(samples[idx], cfg, context_builder, client)
        except Exception:
            logger.exception(
                "%s sample failed. idx=%s sample_key=%s",
                adapter.name,
                idx,
                adapter.sample_key(samples[idx]),
            )
            return None
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
        if get_config_value(cfg, "benchmarks.name") is not None and get_config_value(cfg, "baselines.name") is not None:
            result = _finalize_result_fields(result, cfg=cfg)
        try:
            append_jsonl(result, output_path)
        except Exception:
            logger.exception(
                "Failed to write %s sample JSONL. idx=%s sample_key=%s",
                adapter.name,
                idx,
                adapter.sample_key(samples[idx]),
            )
            return
        samples[idx] = result

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
    if overwrite_enabled(cfg):
        clear_existing_results(output_path)
    compact_results_file(adapter, output_path)
    context_builder = build_context_builder(cfg)
    client = build_openai_client(cfg)

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
        run_pending(adapter, cfg, samples, output_path, context_builder, client)
        finalize_metrics(adapter, cfg, output_path)

    final = finalize_metrics(adapter, cfg, output_path)
    samples = final["samples"]
    metrics = final["metrics"]
    completed = metrics["completed_count"]
    failed = metrics["failed_count"]
    if failed:
        logger.warning("%s stopped with %s/%s successful samples.", adapter.name, completed, len(samples))

    return {
        "results_file": str(output_path),
        "metrics_file": final["metrics_file"],
        "metrics": metrics,
        "samples": samples,
    }
