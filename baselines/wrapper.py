from .image import ImageContextBuilder
from .ocr import OcrContextBuilder

_CONTEXT_BUILDERS = {
    'image': ImageContextBuilder,
    'ocr': OcrContextBuilder,
}


def build_context_builder(cfg):
    if cfg is None:
        raise ValueError('Config is required to build a context_builder.')
    name = cfg.baselines.name
    if name is None:
        raise ValueError('Baseline name not found. Please specify baselines.name in the config.')
    try:
        builder_cls = _CONTEXT_BUILDERS[name]
    except KeyError as exc:
        supported = ', '.join(sorted(_CONTEXT_BUILDERS))
        raise ValueError(f'Unsupported context_builder: {name}. Supported: {supported}') from exc
    return builder_cls(cfg)
