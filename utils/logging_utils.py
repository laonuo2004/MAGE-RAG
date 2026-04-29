import logging


def apply_logging_config(cfg) -> None:
    logging_cfg = cfg.logging

    logging.getLogger().setLevel(logging_cfg.level)
    logging.getLogger("httpx").setLevel(logging_cfg.httpx_level)
    logging.getLogger("httpcore").setLevel(logging_cfg.httpcore_level)
    logging.getLogger("openai").setLevel(logging_cfg.openai_level)
    logging.getLogger("openai._base_client").setLevel(logging_cfg.openai_level)
    logging.getLogger("PIL").setLevel(logging_cfg.pil_level)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging_cfg.pil_pngimageplugin_level)