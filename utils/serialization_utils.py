"""Helpers for converting SDK/config objects into plain data."""

from typing import Any

from omegaconf import OmegaConf


def to_plain_data(value: Any) -> Any:
    """Convert common structured objects into JSON-serializable containers."""
    if value is None:
        return None
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {key: to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_data(item) for item in value]
    return value
