import pkgutil
import importlib
import monitoring.plots
import monitoring.metrics.builders


def load_registers():
    for _, name, _ in pkgutil.iter_modules(monitoring.plots.__path__):
        importlib.import_module(f"monitoring.plots.{name}")

    for _, name, _ in pkgutil.iter_modules(monitoring.metrics.builders.__path__):
        importlib.import_module(f"monitoring.metrics.builders.{name}")
