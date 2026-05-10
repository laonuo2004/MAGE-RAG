import logging

from utils.config_utils import require_config_value
from benchmarks.runner import run_benchmark_with_adapter

logger = logging.getLogger(__name__)


def run_benchmark(cfg):
    benchmark_name = require_config_value(cfg, 'benchmarks.name')
    if benchmark_name == 'mmlongbench':
        from benchmarks.adapters import MMLongBenchAdapter
        logger.info("Running MMLongBench Benchmark")
        return run_benchmark_with_adapter(cfg, MMLongBenchAdapter())
    if benchmark_name == 'longdocurl':
        from benchmarks.adapters import LongDocURLAdapter
        logger.info("Running LongDocURL Benchmark")
        return run_benchmark_with_adapter(cfg, LongDocURLAdapter())
    raise ValueError(f'Unsupported benchmark: {benchmark_name}')
