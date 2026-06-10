from __future__ import annotations

from typing import Any


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def display_score(record: dict[str, Any]) -> float:
    corrected_score = record.get("corrected_score")
    if corrected_score is not None:
        return to_float(corrected_score)
    correction_metadata = record.get("correction_metadata")
    if isinstance(correction_metadata, dict) and correction_metadata.get("corrected_score") is not None:
        return to_float(correction_metadata.get("corrected_score"))
    return to_float(record.get("score"))
