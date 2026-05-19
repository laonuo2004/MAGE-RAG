from .bm25 import BM25ContextBuilder
from .colbertv2 import ColBERTv2ContextBuilder
from .image import ImageContextBuilder
from .ocr import OcrContextBuilder
from .m3docrag import m3docragContextBuilder
from .m3docrag_iterate import M3DocRAGIterateContextBuilder
from utils.config_utils import require_config_value

_CONTEXT_BUILDERS = {
    'image': ImageContextBuilder,
    'ocr': OcrContextBuilder,
    'bm25': BM25ContextBuilder,
    'colbertv2': ColBERTv2ContextBuilder,
    'm3docrag': m3docragContextBuilder,
    'm3docrag-iterate': M3DocRAGIterateContextBuilder,
}


def build_context_builder(cfg):
    if cfg is None:
        raise ValueError('Config is required to build a context_builder.')
    name = require_config_value(cfg, 'baselines.name')
    try:
        builder_cls = _CONTEXT_BUILDERS[name]
    except KeyError as exc:
        supported = ', '.join(sorted(_CONTEXT_BUILDERS))
        raise ValueError(f'Unsupported context_builder: {name}. Supported: {supported}') from exc
    return builder_cls(cfg)
