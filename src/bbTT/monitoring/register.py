from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Set

from bbTT.monitoring.logger.logger import get_logger

logger_inst = get_logger(__name__)


@dataclass
class PlotSpec:
    """
    Args:
        fn (Callable): Plot function, called as fn(ctx, **kwargs).
        requires (set[str]): Context artifacts the plot needs, resolved through the builders.
        kwargs (dict): Extra keyword arguments of the plot, the way a variant is parametrised.
        optional (bool): Skip silently when a requirement cannot be met.
        tag (str, optional): Tensorboard tag of the figure. None uses the registry name. Set it to place the
            figure next to related scalars, e.g. "MonitoredBlocks/input_layer/angle_centered".
    """
    fn: Callable
    requires: Set[str] = None
    kwargs: dict = field(default_factory=dict)
    optional: bool = False
    tag: str | None = None


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
def register_builder(name, *, requires=None, provides=None):
    requires = set(requires or [])
    provides = set(provides or [])

    for existing in BUILDER_REGISTRY.values():
        overlap = existing.provides & provides
        if overlap:
            raise ValueError(f"Builder output(s) already registered: {overlap}")

    # wrapper needs to have **kwargs
    # so it can access requires and provides from decorator
    def wrapper(fn):
        if name in BUILDER_REGISTRY:
            logger_inst.warning(f"Trying to register builder: {name}, but already exist")

        BUILDER_REGISTRY[name] = BuilderSpec(
            fn=fn,
            requires=requires,
            provides=provides,
        )

        return fn

    return wrapper


def register_plot(name, requires=None, optional=False, tag=None, **kwargs):

    # kwargs contain extra arguments passed to the decorated function
    def wrapper(fn):
        if name in PLOT_REGISTRY:
            logger_inst.warning(f"Trying to register plot: {name}, but already exist")

        PLOT_REGISTRY[name] = PlotSpec(
            fn=fn,
            requires=set(requires or []),
            optional=optional,
            kwargs=kwargs,
            tag=tag,

        )
        return fn

    return wrapper


def register_plot_variant(name, base, optional=None, tag=None, **kwargs):
    spec = PLOT_REGISTRY[base]
    PLOT_REGISTRY[name] = PlotSpec(
        fn=spec.fn,
        requires=spec.requires,
        kwargs=kwargs,
        # variant can override optionality and tag, otherwise inherits from base
        optional=optional if optional is not None else spec.optional,
        tag=tag if tag is not None else spec.tag,
    )


def register_scalar(name, requires=None):
    def wrapper(fn):
        SCALAR_REGISTRY[name] = ScalarSpec(fn=fn, requires=set(requires or []))
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
