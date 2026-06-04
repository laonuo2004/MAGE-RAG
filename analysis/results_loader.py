from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from analysis.plugins import ParameterSpec, RunContext, get_plugin, registered_plugins


@dataclass(frozen=True)
class RunRecord:
    benchmark: str
    baseline: str
    run_id: str
    stem: str
    jsonl_path: Path | None
    metrics_path: Path | None
    jsonl_size_bytes: int | None
    metrics_size_bytes: int | None
    sample_count: int | None
    completed_count: int | None
    failed_count: int | None
    parameters: dict[str, Any]
    metrics: dict[str, Any]
    plugin_name: str = "default"
    parameter_sources: dict[str, str] | None = None


def parse_run_parameters(
    path: Path | str,
    parameter_specs: Iterable[ParameterSpec] | None = None,
) -> dict[str, Any]:
    """Extract declared run parameters from legacy result filenames."""
    specs = tuple(parameter_specs) if parameter_specs is not None else _all_known_parameter_specs()
    if not specs:
        return {}

    name = Path(path).name
    stem = name.removesuffix(".metrics.json")
    stem = stem.removesuffix(".jsonl")
    tokens = stem.split("_")
    params: dict[str, Any] = {}

    index = 0
    while index < len(tokens):
        matched = False
        for spec in specs:
            key_tokens = spec.name.split("_")
            next_index = index + len(key_tokens)
            if tokens[index:next_index] != key_tokens or next_index >= len(tokens):
                continue
            value, consumed = _consume_parameter_value(tokens, next_index, spec, specs)
            if consumed > next_index:
                params[spec.name] = value
                index = consumed
                matched = True
                break
        if not matched:
            index += 1
    return params


def scan_runs(results_root: Path | str = "results") -> list[RunRecord]:
    root = Path(results_root)
    grouped: dict[tuple[str, str, str], dict[str, Path]] = {}
    for path in root.glob("*/*/*.jsonl"):
        key = _run_key(root, path, ".jsonl")
        grouped.setdefault(key, {})["jsonl"] = path
    for path in root.glob("*/*/*.metrics.json"):
        key = _run_key(root, path, ".metrics.json")
        grouped.setdefault(key, {})["metrics"] = path

    records = []
    for (benchmark, baseline, stem), paths in sorted(grouped.items()):
        metrics_path = paths.get("metrics")
        jsonl_path = paths.get("jsonl")
        metrics = _read_json(metrics_path) if metrics_path else {}
        first_record = _read_first_jsonl_record(jsonl_path) if jsonl_path else None
        plugin = get_plugin(baseline)
        context = RunContext(
            benchmark=benchmark,
            baseline=baseline,
            stem=stem,
            jsonl_path=jsonl_path,
            metrics_path=metrics_path,
            metrics=metrics,
            first_record=first_record,
        )
        params, sources = plugin.extract_parameters(context)
        filename_params = parse_run_parameters(jsonl_path or metrics_path or Path(stem), plugin.parameter_specs())
        for key, value in filename_params.items():
            if key not in params:
                params[key] = value
                sources[key] = "filename"

        records.append(
            RunRecord(
                benchmark=benchmark,
                baseline=baseline,
                run_id=f"{benchmark}/{baseline}/{stem}",
                stem=stem,
                jsonl_path=jsonl_path,
                metrics_path=metrics_path,
                jsonl_size_bytes=_file_size(jsonl_path),
                metrics_size_bytes=_file_size(metrics_path),
                sample_count=_optional_int(metrics.get("sample_count")),
                completed_count=_optional_int(metrics.get("completed_count")),
                failed_count=_optional_int(metrics.get("failed_count")),
                parameters=params,
                metrics=metrics,
                plugin_name=plugin.name,
                parameter_sources=sources,
            )
        )
    return records


def read_jsonl_cached(
    path: Path | str,
    cache_root: Path | str = "analysis_cache/result_analysis",
    cache_namespace: str = "default",
) -> list[dict[str, Any]]:
    source = Path(path)
    stat = source.stat()
    cache_dir = Path(cache_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{_cache_key(source, cache_namespace)}.json"
    signature = {
        "path": str(source.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "fingerprint": _file_fingerprint(source, stat.st_size),
        "cache_namespace": cache_namespace,
    }

    if cache_path.exists():
        cached = _read_json(cache_path)
        if cached.get("signature") == signature:
            return cached.get("records", [])

    records = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    cache_path.write_text(
        json.dumps({"signature": signature, "records": records}, ensure_ascii=False),
        encoding="utf-8",
    )
    return records


def enrich_parameters_from_records(
    parameters: dict[str, Any],
    records: list[dict[str, Any]],
    parameter_specs: Iterable[ParameterSpec] | None = None,
) -> dict[str, Any]:
    enriched = dict(parameters)
    if not records:
        return enriched
    metadata = records[0].get("prepare_metadata") or {}
    specs = tuple(parameter_specs) if parameter_specs is not None else _all_known_parameter_specs()
    for spec in specs:
        if spec.name not in enriched and spec.name in metadata:
            enriched[spec.name] = metadata[spec.name]
    return enriched


def _run_key(root: Path, path: Path, suffix: str) -> tuple[str, str, str]:
    relative = path.relative_to(root)
    benchmark = relative.parts[0]
    baseline = relative.parts[1]
    stem = path.name.removesuffix(suffix)
    return benchmark, baseline, stem


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_first_jsonl_record(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    return None


def _file_size(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    return path.stat().st_size


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _consume_parameter_value(
    tokens: list[str],
    start: int,
    spec: ParameterSpec,
    specs: tuple[ParameterSpec, ...],
) -> tuple[Any, int]:
    if start >= len(tokens):
        return None, start
    if spec.value_type == "bool":
        return _coerce_scalar(tokens[start]), start + 1
    if spec.value_type == "int":
        token = tokens[start]
        if token.isdigit():
            return int(token), start + 1
        return None, start
    if spec.value_type == "float":
        parts, consumed = _consume_numeric_parts(tokens, start, specs)
        if not parts:
            return None, start
        return _coerce_number_parts(parts), consumed
    if spec.value_type == "str":
        parts: list[str] = []
        index = start
        while index < len(tokens):
            if _starts_known_key(tokens, index, specs) and parts:
                break
            parts.append(tokens[index])
            index += 1
        return "_".join(parts), index

    token = tokens[start]
    if token in {"None", "True", "False"}:
        return _coerce_scalar(token), start + 1
    parts, consumed = _consume_numeric_parts(tokens, start, specs)
    if parts:
        return _coerce_number_parts(parts), consumed
    return token, start + 1


def _consume_numeric_parts(
    tokens: list[str],
    start: int,
    specs: tuple[ParameterSpec, ...],
) -> tuple[list[str], int]:
    parts: list[str] = []
    index = start
    while index < len(tokens):
        if _starts_known_key(tokens, index, specs) and parts:
            break
        token = tokens[index]
        if token.isdigit():
            parts.append(token)
            index += 1
            continue
        break
    return parts, index


def _starts_known_key(tokens: list[str], index: int, specs: tuple[ParameterSpec, ...]) -> bool:
    return any(tokens[index : index + len(spec.name.split("_"))] == spec.name.split("_") for spec in specs)


def _coerce_scalar(value: str) -> Any:
    if value == "True":
        return True
    if value == "False":
        return False
    if value == "None":
        return None
    return value


def _coerce_number_parts(parts: list[str]) -> int | float:
    if len(parts) == 1:
        return int(parts[0])
    return float(f"{parts[0]}.{''.join(parts[1:])}")


def _all_known_parameter_specs() -> tuple[ParameterSpec, ...]:
    specs: dict[str, ParameterSpec] = {}
    for plugin in registered_plugins():
        for spec in plugin.parameter_specs():
            specs.setdefault(spec.name, spec)
    return tuple(specs.values())


def _file_fingerprint(path: Path, size: int, sample_bytes: int = 65536) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        hasher.update(handle.read(sample_bytes))
        if size > sample_bytes:
            handle.seek(max(size - sample_bytes, 0))
            hasher.update(handle.read(sample_bytes))
    return hasher.hexdigest()


def _cache_key(path: Path, namespace: str) -> str:
    return hashlib.sha256(f"{namespace}:{path.resolve()}".encode("utf-8")).hexdigest()
