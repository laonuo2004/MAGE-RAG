import hydra
import logging
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from benchmarks.wrapper import run_benchmark


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Set the logging level based on the configuration
    logging.getLogger().setLevel(cfg.logging.level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Benchmark With The Following Configuration:")
    logger.info("\n" + OmegaConf.to_yaml(cfg, resolve=True))
    
    run_benchmark(cfg)

    # Useful when checking where Hydra stores per-run artifacts.
    logger.info(f"Hydra Output Dir: {HydraConfig.get().runtime.output_dir}")

if __name__ == "__main__":
    main()
