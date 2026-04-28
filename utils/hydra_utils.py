import hydra
from omegaconf import DictConfig, OmegaConf

def _value(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)