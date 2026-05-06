import logging

from utils.config_utils import require_config_value


def apply_logging_config(cfg) -> None:
    logging_cfg = require_config_value(cfg, 'logging')

    logging.getLogger().setLevel(require_config_value(logging_cfg, 'level'))
    logging.getLogger("httpx").setLevel(require_config_value(logging_cfg, 'httpx_level'))
    logging.getLogger("httpcore").setLevel(require_config_value(logging_cfg, 'httpcore_level'))
    logging.getLogger("openai").setLevel(require_config_value(logging_cfg, 'openai_level'))
    logging.getLogger("openai._base_client").setLevel(require_config_value(logging_cfg, 'openai_level'))
    logging.getLogger("PIL").setLevel(require_config_value(logging_cfg, 'pil_level'))
    logging.getLogger("PIL.PngImagePlugin").setLevel(require_config_value(logging_cfg, 'pil_pngimageplugin_level'))
