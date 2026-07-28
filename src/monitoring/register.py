from dataclasses import dataclass, field
from typing import Callable, Set

import warnings


@dataclass
class PlotSpec:
    fn: Callable
    requires: Set[str] = None
    kwargs: dict = field(default_factory=dict)

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
        if name in BUILDER_REGISTRY:
            warnings.warn(f"Trying to register builder: {name}, but already exist")

        BUILDER_REGISTRY[name] = BuilderSpec(
            fn=fn,
            requires=requires,
            provides=provides,
        )

        return fn
    return wrapper


def register_plot(name, requires=None, **kwargs):

    # kwargs contain extra arguments passed to the decorated function
    def wrapper(fn):
        if name in PLOT_REGISTRY:
            warnings.warn(f"Trying to register plot: {name}, but already exist")


        PLOT_REGISTRY[name] = PlotSpec(
            fn=fn,
            requires=set(requires or []),
            kwargs=kwargs,
        )
        return fn
    return wrapper

def register_plot_variant(
    name,
    base,
    **kwargs,
):
    spec = PLOT_REGISTRY[base]
    PLOT_REGISTRY[name] = PlotSpec(
        fn=spec.fn,
        requires=spec.requires,
        kwargs=kwargs,
    )

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
