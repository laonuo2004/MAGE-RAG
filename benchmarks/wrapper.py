import sys
from argparse import Namespace
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from omegaconf import OmegaConf
from utils.hydra_utils import _value

import logging
logger = logging.getLogger(__name__)

BENCHMARK_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(BENCHMARK_ROOT))  

def _run_mmlongbench(cfg):
    from mmlongbench.run_api import run_mmlongbench

    # provider_name = _provider_name_for_benchmark(benchmark_cfg, default='litellm')
    # provider = _provider_cfg(cfg, provider_name)
    # args = _namespace_from_dict({
    #     'input_path': _value(benchmark_cfg, 'input_path'),
    #     'document_path': _value(benchmark_cfg, 'document_path'),
    #     'model_name': _resolve_model_name(cfg, benchmark_cfg, provider_name=provider_name),
    #     'api_style': _value(benchmark_cfg, 'api_style', 'openai'),
    #     'base_url': _value(benchmark_cfg, 'base_url', provider.get('base_url')),
    #     'api_key': _value(benchmark_cfg, 'api_key', provider.get('api_key')),
    #     'extractor_model_name': _value(benchmark_cfg, 'extractor_model_name'),
    #     'extractor_base_url': _value(benchmark_cfg, 'extractor_base_url'),
    #     'extractor_api_key': _value(benchmark_cfg, 'extractor_api_key'),
    #     'max_pages': _value(benchmark_cfg, 'max_pages'),
    #     'resolution': _value(benchmark_cfg, 'resolution'),
    #     'max_try': _value(benchmark_cfg, 'max_try'),
    #     'max_tokens': _value(benchmark_cfg, 'max_tokens'),
    #     'temperature': _value(benchmark_cfg, 'temperature'),
    #     'extractor_prompt_path': _value(benchmark_cfg, 'extractor_prompt_path'),
    #     'output_path': _value(benchmark_cfg, 'output_path'),
    #     'results_dir': _value(benchmark_cfg, 'results_dir', str(BENCHMARK_ROOT / 'mmlongbench' / 'results')),
    #     'tmp_dir': _value(benchmark_cfg, 'tmp_dir', str(BENCHMARK_ROOT / 'mmlongbench' / 'tmp')),
    #     'limit': _value(benchmark_cfg, 'limit'),
    #     'sample_id': _value(benchmark_cfg, 'sample_id'),
    #     'sample_doc_id': _value(benchmark_cfg, 'sample_doc_id'),
    #     'sample_question': _value(benchmark_cfg, 'sample_question'),
    #     'num_workers': _value(benchmark_cfg, 'num_workers'),
    #     'route_base_urls': _value(benchmark_cfg, 'route_base_urls'),
    #     'route_model_names': _value(benchmark_cfg, 'route_model_names'),
    #     'route_api_keys': _value(benchmark_cfg, 'route_api_keys'),
    #     'route_labels': _value(benchmark_cfg, 'route_labels'),
    #     'route_max_model_lens': _value(benchmark_cfg, 'route_max_model_lens'),
    #     'debug_prompts': _value(benchmark_cfg, 'debug_prompts', False),
    #     'debug_dir': _value(benchmark_cfg, 'debug_dir'),
    #     'context_builder': context_builder_name,
    #     'baselines': {'name': context_builder_name},
    # })
    return run_mmlongbench(cfg)
    

def _run_longdocurl(cfg):
    from longdocurl.eval.api_models.eval_api_models import run_longdocurl

    # llm_provider = _value(benchmark_cfg, 'llm_provider', 'openrouter')
    # model_name = _resolve_model_name(cfg, benchmark_cfg, provider_name=llm_provider)
    # args = _namespace_from_dict({
    #     'qa_file': _value(benchmark_cfg, 'qa_file'),
    #     'process_mode': _value(benchmark_cfg, 'process_mode'),
    #     'workers': _value(benchmark_cfg, 'workers'),
    #     'input_format': 'ocr' if context_builder_name == 'ocr' else 'e2e',
    #     'context_builder': context_builder_name,
    #     'ocr_backend': _value(benchmark_cfg, 'ocr_backend'),
    #     'ocr_json_dir': _value(benchmark_cfg, 'ocr_json_dir'),
    #     'image_prefix': _value(benchmark_cfg, 'image_prefix'),
    #     'qa_llm_provider': _value(benchmark_cfg, 'qa_llm_provider', llm_provider),
    #     'extractor_llm_provider': _value(benchmark_cfg, 'extractor_llm_provider', llm_provider),
    #     'qa_model_name': _value(benchmark_cfg, 'qa_model_name', model_name),
    #     'extractor_model_name': _value(benchmark_cfg, 'extractor_model_name', model_name),
    #     'results_file': _value(benchmark_cfg, 'results_file'),
    # })
    return run_longdocurl(cfg)

def run_benchmark(cfg):
    benchmark_cfg = _value(cfg, 'benchmarks', {})
    benchmark_name = _value(benchmark_cfg, 'name', None)
    # context_builder_name = _context_builder_name(cfg)
    if benchmark_name == 'mmlongbench':
        logger.info("Running MMLongBench benchmark")
        return _run_mmlongbench(cfg)
    if benchmark_name == 'longdocurl':
        logger.info("Running LongDocURL benchmark")
        return _run_longdocurl(cfg)
    raise ValueError(f'Unsupported benchmark: {benchmark_name}')