import hydra
import logging
from pathlib import Path

from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from baselines.bgem3_cache import prepare_bgem3_cache
from baselines.colbertv2_cache import prepare_colbertv2_cache
from benchmarks.wrapper import run_benchmark
from utils.logging_utils import apply_logging_config


load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")


def prepare_baseline_cache(cfg):
    baseline_name = cfg.baselines.name
    if baseline_name == "colbertv2":
        prepare_colbertv2_cache(cfg)
        return
    if baseline_name == "bgem3":
        prepare_bgem3_cache(cfg)
        return


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    apply_logging_config(cfg)

    logger = logging.getLogger(__name__)
    logger.info("Starting Benchmark With The Following Configuration:")
    logger.info("\n" + OmegaConf.to_yaml(cfg, resolve=True))

    prepare_baseline_cache(cfg)
    run_benchmark(cfg)

    logger.info(f"Hydra Output Dir: {HydraConfig.get().runtime.output_dir}")


if __name__ == "__main__":
    main()
