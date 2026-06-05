from baselines.bgem3 import BGEM3ContextBuilder
from baselines.bm25 import BM25ContextBuilder
from baselines.colbertv2 import ColBERTv2ContextBuilder
from baselines.image import ImageContextBuilder
from baselines.ocr import OcrContextBuilder
from baselines.m3docrag import m3docragContextBuilder
from baselines.m3docrag_iterate import M3DocRAGIterateContextBuilder
from baselines.m3docrag_iterate_query import M3DocRAGIterateQueryContextBuilder
from baselines.magerag import MAGERAGContextBuilder
from baselines.g2reader import G2ReaderContextBuilder
from utils.config_utils import require_config_value

_CONTEXT_BUILDERS = {
    'image': ImageContextBuilder,
    'ocr': OcrContextBuilder,
    'bgem3': BGEM3ContextBuilder,
    'bm25': BM25ContextBuilder,
    'colbertv2': ColBERTv2ContextBuilder,
    'm3docrag': m3docragContextBuilder,
    'm3docrag-iterate': M3DocRAGIterateContextBuilder,
    'm3docrag-iterate-query': M3DocRAGIterateQueryContextBuilder,
    'magerag': MAGERAGContextBuilder,
    'g2-reader': G2ReaderContextBuilder,
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
