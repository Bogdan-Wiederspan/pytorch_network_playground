import importlib
import pkgutil

import bbTT.monitoring.metrics.builders
import bbTT.monitoring.plots


def load_registers():
    # function to load all builders and plots to register them in the registries
    for _, name, _ in pkgutil.iter_modules(bbTT.monitoring.plots.__path__):
        importlib.import_module(f"bbTT.monitoring.plots.{name}")

    for _, name, _ in pkgutil.iter_modules(bbTT.monitoring.metrics.builders.__path__):
        importlib.import_module(f"bbTT.monitoring.metrics.builders.{name}")
