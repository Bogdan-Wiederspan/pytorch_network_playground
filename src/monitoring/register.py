from dataclasses import dataclass
from typing import Callable, Set

@dataclass
class SignalSpec:
    fn: Callable
    requires: Set[str] = None
    provides: Set[str] = None
    kind: Set[str] = None

@dataclass
class PlotSpec:
    fn: Callable
    requires: Set[str] = None

@dataclass
class BuilderSpec:
    fn: Callable
    provides: Set[str] = None
    requires: Set[str] = None

@dataclass
class ScalarSpec:
    fn: callable
    requires: Set[str] = None
    provides: Set[str] = None


PLOT_REGISTRY: dict[str, PlotSpec] = {}
BUILDER_REGISTRY: dict[str, BuilderSpec] = {}
SCALAR_REGISTRY: dict[str, ScalarSpec] = {}
SIGNAL_REGISTRY: dict[str, SignalSpec] = {}

# builders need to return values as dictionary with provides keys as keys
def register_builder(name, * ,requires=None, provides=None):
    requires = set(requires or [])
    provides = set(provides or [])

    for existing in BUILDER_REGISTRY.values():
        overlap = existing.provides & provides
        if overlap:
            raise ValueError(
                f"Builder output(s) already registered: {overlap}"
            )

    # wrapper needs to have **kwargs
    # so it can access requires and provides from decorator
    def wrapper(fn):
        BUILDER_REGISTRY[name] = BuilderSpec(
            fn=fn,
            requires=requires,
            provides=provides,
        )

        return fn
    return wrapper


def register_plot(name, requires=None):
    def wrapper(fn):
        PLOT_REGISTRY[name] = PlotSpec(
            fn=fn,
            requires=set(requires or [])
        )
        return fn
    return wrapper

def register_scalar(name, requires=None):
    def wrapper(fn):
        SCALAR_REGISTRY[name] = ScalarSpec(
            fn=fn,
            requires=set(requires or [])
        )
        return fn
    return wrapper

class PlotContext:
    def __init__(self, pred, target, weights, target_map, data):
        self.pred = pred
        self.target = target
        self.weights = weights
        self.target_map = target_map
        self.data = data

    def has(self, key):
        return key in self.data

    def get(self, key, default=None):
        return self.data.get(key, default)
