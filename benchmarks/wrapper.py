import logging

from utils.config_utils import require_config_value

logger = logging.getLogger(__name__)

def _run_mmlongbench(cfg):
    from benchmarks.mmlongbench.run_api import run_mmlongbench
    return run_mmlongbench(cfg)
    

def _run_longdocurl(cfg):
    from benchmarks.longdocurl.eval.api_models.eval_api_models import run_longdocurl
    return run_longdocurl(cfg)

def run_benchmark(cfg):
    benchmark_name = require_config_value(cfg, 'benchmarks.name')
    if benchmark_name == 'mmlongbench':
        logger.info("Running MMLongBench Benchmark")
        return _run_mmlongbench(cfg)
    if benchmark_name == 'longdocurl':
        logger.info("Running LongDocURL Benchmark")
        return _run_longdocurl(cfg)
    raise ValueError(f'Unsupported benchmark: {benchmark_name}')
