import hydra
import logging
from pathlib import Path

from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from benchmarks.utils.embedding_cache import prepare_embedding_cache
from benchmarks.wrapper import run_benchmark
from utils.logging_utils import apply_logging_config


load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    apply_logging_config(cfg)

    logger = logging.getLogger(__name__)
    logger.info("Starting Benchmark With The Following Configuration:")
    logger.info("\n" + OmegaConf.to_yaml(cfg, resolve=True))

    prepare_embedding_cache(cfg)
    run_benchmark(cfg)

    logger.info(f"Hydra Output Dir: {HydraConfig.get().runtime.output_dir}")


if __name__ == "__main__":
    main()
