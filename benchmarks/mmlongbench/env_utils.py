import os
from pathlib import Path


def load_env_file(env_path):
    env_vars = {}
    path = Path(env_path)
    if not path.exists():
        return env_vars

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        env_vars[key] = value
    return env_vars


def load_local_env():
    merged = {}
    for name in (".env.mmlongbench", ".env"):
        merged.update(load_env_file(name))
    return merged


def get_config_value(cli_value, env_name, local_env=None, default=None):
    if cli_value is not None:
        return cli_value
    if env_name in os.environ:
        return os.environ[env_name]
    if local_env and env_name in local_env:
        return local_env[env_name]
    return default
