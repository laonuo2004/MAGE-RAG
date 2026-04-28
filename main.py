import hydra
import logging
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from benchmarks.wrapper import run_benchmark


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Starting benchmark with the following configuration:")
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    
    run_benchmark(cfg)

    # Useful when checking where Hydra stores per-run artifacts.
    logger.info(f"Hydra output dir: {HydraConfig.get().runtime.output_dir}")

if __name__ == "__main__":
    main()
