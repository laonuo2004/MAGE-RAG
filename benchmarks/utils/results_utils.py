import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from utils.config_utils import get_config_value, require_config_value

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"


def sanitize_name(value: Any) -> str:
    text = str(value)
    text = text.replace(":free", "")
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def result_param_parts(cfg) -> List[Tuple[str, Any]]:
    params = get_config_value(cfg, "baselines.params")
    if params is None:
        return []
    return list(params.items())


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
