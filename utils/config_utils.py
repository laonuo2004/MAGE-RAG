"""Helpers for reading Hydra/OmegaConf configuration values."""

from omegaconf import OmegaConf

_MISSING = object()


def _select_from_mapping_or_object(cfg, key, default):
    current = cfg
    for part in str(key).split("."):
        if current is None:
            return default
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
        elif hasattr(current, "get"):
            value = current.get(part, _MISSING)
            if value is _MISSING:
                return default
            current = value
        else:
            if not hasattr(current, part):
                return default
            current = getattr(current, part)
    return current


def get_config_value(cfg, key, default=None, *, required=False):
    """Read a Hydra/OmegaConf config value with one project-wide convention.

    ``key`` may be a dotted path, for example ``"benchmarks.process_mode"``.
    Use ``required=True`` for values that must be present and non-None.
    """
    if key in (None, ""):
        value = cfg
    else:
        try:
            value = OmegaConf.select(cfg, key, default=_MISSING)
        except Exception:
            value = _select_from_mapping_or_object(cfg, key, _MISSING)

    if value is _MISSING or (required and value is None):
        if required:
            raise ValueError(f"Missing required config value: {key}")
        return default
    return value


def require_config_value(cfg, key):
    """Read a required Hydra/OmegaConf config value."""
    return get_config_value(cfg, key, required=True)
