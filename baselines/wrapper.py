import sys
import pathlib

from omegaconf import OmegaConf

from .image import ImageContextBuilder
from .ocr import OcrContextBuilder

_CONTEXT_BUILDERS = {
    'image': ImageContextBuilder,
    'ocr': OcrContextBuilder,
}


# def _to_plain_container(cfg):
#     if cfg is None:
#         return {}
#     if isinstance(cfg, (dict, list, tuple, str, int, float, bool)):
#         return cfg
#     if not OmegaConf.is_config(cfg):
#         return cfg
#     return OmegaConf.to_container(cfg, resolve=True)


def build_context_builder(cfg):
    # baseline_cfg = getattr(cfg, 'baselines', cfg)
    # baseline_cfg = _to_plain_container(baseline_cfg)
    baseline_cfg = cfg.get('baselines', {})
    name = baseline_cfg.name
    if name is None:
        raise ValueError('Baseline name not found. Please specify baselines.name in the config.')
    try:
        builder_cls = _CONTEXT_BUILDERS[name]
    except KeyError as exc:
        supported = ', '.join(sorted(_CONTEXT_BUILDERS))
        raise ValueError(f'Unsupported context_builder: {name}. Supported: {supported}') from exc
    return builder_cls(baseline_cfg)
