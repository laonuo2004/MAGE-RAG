from __future__ import annotations

import importlib
import pkgutil
from functools import lru_cache

from analysis.plugins.base import AnalysisPlugin, ChartSpec, ParameterSpec, RunContext


@lru_cache(maxsize=1)
def registered_plugins() -> tuple[AnalysisPlugin, ...]:
    plugins: list[AnalysisPlugin] = []
    package_name = __name__
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name in {"base"}:
            continue
        module = importlib.import_module(f"{package_name}.{module_info.name}")
        plugin = getattr(module, "PLUGIN", None)
        if isinstance(plugin, AnalysisPlugin):
            plugins.append(plugin)
        get_plugins = getattr(module, "get_plugins", None)
        if callable(get_plugins):
            for item in get_plugins():
                if isinstance(item, AnalysisPlugin):
                    plugins.append(item)
    return tuple(plugins)


def default_plugin() -> AnalysisPlugin:
    from analysis.plugins.builtin import DefaultPlugin

    return DefaultPlugin()


def get_plugin(baseline: str) -> AnalysisPlugin:
    for plugin in registered_plugins():
        if plugin.matches(baseline):
            return plugin
    return default_plugin()


def parameter_specs_for_baseline(baseline: str) -> tuple[ParameterSpec, ...]:
    return get_plugin(baseline).parameter_specs()


__all__ = [
    "AnalysisPlugin",
    "ChartSpec",
    "ParameterSpec",
    "RunContext",
    "default_plugin",
    "get_plugin",
    "parameter_specs_for_baseline",
    "registered_plugins",
]
