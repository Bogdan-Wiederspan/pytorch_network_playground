import importlib
import pkgutil

import monitoring.metrics.builders
import monitoring.plots


def load_registers():
    # function to load all builders and plots to register them in the registries
    for _, name, _ in pkgutil.iter_modules(monitoring.plots.__path__):
        importlib.import_module(f"monitoring.plots.{name}")

    for _, name, _ in pkgutil.iter_modules(monitoring.metrics.builders.__path__):
        importlib.import_module(f"monitoring.metrics.builders.{name}")
