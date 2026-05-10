import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from omegaconf import OmegaConf

from utils.config_utils import get_config_value, require_config_value

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"


def sanitize_name(value: Any) -> str:
    text = str(value)
    text = text.replace(":free", "")
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def result_param_parts(cfg) -> List[Tuple[str, Any]]:
    baseline_name = require_config_value(cfg, "baselines.name")
    if baseline_name == "m3docrag":
        return [("top_k", int(require_config_value(cfg, "baselines.top_k")))]
    if baseline_name == "bm25":
        return [
            ("top_k", int(require_config_value(cfg, "baselines.top_k"))),
            ("chunk_size", int(require_config_value(cfg, "baselines.chunk_size"))),
            ("chunk_overlap", int(require_config_value(cfg, "baselines.chunk_overlap"))),
        ]
    return []


def build_results_file(cfg) -> Path:
    benchmark_name = require_config_value(cfg, "benchmarks.name")
    baseline_name = require_config_value(cfg, "baselines.name")
    qa_model_name = require_config_value(cfg, "benchmarks.qa_model_name")
    params = result_param_parts(cfg)
    parts = ["res"]
    for key, value in params:
        parts.extend([key, sanitize_name(value)])
    parts.append(sanitize_name(qa_model_name))
    filename = "_".join(parts) + ".jsonl"
    return RESULTS_ROOT / sanitize_name(benchmark_name) / sanitize_name(baseline_name) / filename


def sidecar_paths(results_file: str | Path) -> Tuple[Path, Path]:
    path = Path(results_file)
    return path.with_suffix(".metrics.json"), path.with_suffix(".manifest.json")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    results = []
    if not path.exists():
        return results
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSONL line in %s:%s: %s", path, line_no, exc)
    return results


def append_jsonl(record: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def config_summary(cfg) -> Dict[str, Any]:
    try:
        resolved = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        resolved = {}
    return {
        "benchmarks": resolved.get("benchmarks", {}) if isinstance(resolved, dict) else {},
        "baselines": resolved.get("baselines", {}) if isinstance(resolved, dict) else {},
        "litellm": {
            "base_url": get_config_value(cfg, "litellm.base_url"),
        },
    }


def build_manifest(cfg, results_file: str | Path, metrics_file: str | Path) -> Dict[str, Any]:
    params = {key: value for key, value in result_param_parts(cfg)}
    return {
        "generated_at": utc_now(),
        "benchmark": require_config_value(cfg, "benchmarks.name"),
        "baseline": require_config_value(cfg, "baselines.name"),
        "qa_model_name": require_config_value(cfg, "benchmarks.qa_model_name"),
        "extractor_model_name": get_config_value(cfg, "benchmarks.extractor_model_name"),
        "parameters": params,
        "results_file": str(Path(results_file)),
        "metrics_file": str(Path(metrics_file)),
        "resolved_config": config_summary(cfg),
    }


def count_jsonl(path: str | Path) -> int:
    return sum(1 for _ in read_jsonl(path))


def iter_result_manifests(root: str | Path = RESULTS_ROOT) -> Iterable[Tuple[Path, Dict[str, Any], Dict[str, Any] | None]]:
    root = Path(root)
    for manifest_path in sorted(root.glob("*/*/*.manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping invalid manifest %s: %s", manifest_path, exc)
            continue
        metrics_path = Path(manifest.get("metrics_file") or manifest_path.with_suffix("").with_suffix(".metrics.json"))
        metrics = None
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Skipping invalid metrics %s: %s", metrics_path, exc)
        yield manifest_path, manifest, metrics
