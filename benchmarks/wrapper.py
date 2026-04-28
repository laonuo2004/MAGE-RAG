import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

import logging
logger = logging.getLogger(__name__)

BENCHMARK_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(BENCHMARK_ROOT)) 

def _run_mmlongbench(cfg):
    from mmlongbench.run_api import run_mmlongbench
    return run_mmlongbench(cfg)
    

def _run_longdocurl(cfg):
    from longdocurl.eval.api_models.eval_api_models import run_longdocurl
    return run_longdocurl(cfg)

def run_benchmark(cfg):
    benchmark_cfg = cfg.get('benchmarks', {})
    benchmark_name = benchmark_cfg.get('name', None)
    if benchmark_name == 'mmlongbench':
        logger.info("Running MMLongBench Benchmark")
        return _run_mmlongbench(cfg)
    if benchmark_name == 'longdocurl':
        logger.info("Running LongDocURL Benchmark")
        return _run_longdocurl(cfg)
    raise ValueError(f'Unsupported benchmark: {benchmark_name}')
