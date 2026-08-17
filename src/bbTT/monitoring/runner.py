from bbTT.monitoring.logger.logger import get_logger
from bbTT.monitoring.register import BUILDER_REGISTRY, PLOT_REGISTRY

_PROVIDER_MAP_CACHE = None



logger_inst = get_logger(__name__)

def require_map(registry):
    _map = {}
    for name, spec in registry.items():
        requirements = spec.requires
        if isinstance(requirements, str):
            requirements = [requirements]
        for requirement in requirements:
            if requirement not in _map:
                _map[requirement] = []
            _map[requirement].append(name)
    return _map

def build_provider_map(force_refresh=False):
    # static resolution of provider map
    # NO DYNAMIC SUPPORT due to caching
    global _PROVIDER_MAP_CACHE
    if _PROVIDER_MAP_CACHE is None or force_refresh:
        _PROVIDER_MAP_CACHE = {
            artifact : name
            for name, spec in BUILDER_REGISTRY.items()
            for artifact in spec.provides
        }
    return _PROVIDER_MAP_CACHE


def ensure(ctx, artifact, providers, _resolving=None, requester=None):
    # check if artifact is already provided otherwise build it via builder
    # for builder ensure that their dependencies also exist

    if _resolving is None:
        _resolving = set()

    # --- resolving patterns ---
    concrete_keys = ctx.expand(artifact)

    if not concrete_keys:
            raise RequirementNotMet(artifact, requester)


    # if the pattern expanded to multiple keys, ensure each one
    if len(concrete_keys) > 1 or concrete_keys[0] != artifact:
        for key in concrete_keys:
            ensure(ctx, key, providers, _resolving, requester=requester)
        return

    # --- single concrete key from here ---

    if ctx.has(artifact):
        return

    if artifact in _resolving:
        raise ValueError(f"Circular dependency detected while resolving in {artifact}")

    builder_name = providers.get(artifact)

    if builder_name is None:

        # No builder exists AND not in context — this is the "skip" case
        raise RequirementNotMet(artifact, requester)

    builder = BUILDER_REGISTRY[builder_name]
    _resolving.add(artifact)

    for dep in builder.requires:
        ensure(ctx, dep, providers, _resolving, requester=requester)

    result = builder.fn(ctx)
    # check if builder really provided something and did not silently died
    missing_provides = builder.provides - result.keys()
    if missing_provides:
        raise ValueError(
            f"Builder {builder_name} should provide {builder.provides}"
            f"but did not return: {missing_provides}"
        )
    # save result in cache of context
    ctx.cache.update(result)

def run_plot(name, ctx):
    spec = PLOT_REGISTRY[name]

    providers = build_provider_map()

    for req in spec.requires:
        ensure(ctx, req, providers, requester=name)
    return spec.fn(ctx)


def run_scalar(name, ctx):
    # scalars
    providers = build_provider_map()
    ensure(ctx, name, providers, requester=name)
    return ctx.get(name)


class EvaluationRunner:

    def __init__(self, tensorboard):
        """
        Actual runner that gets a list of plots, meta data and context object that
        holds the data to plot and saves these in its corresponding tensorboard instance.

        Args:
            tensorboard (torch.Tensorboard): Tensorboard instance, where everything is logged.
        """
        self.tensorboard = tensorboard
        self._skipped_plots: set[str, str] = set() # plots skipped due to being optional or not registered, second str is the mode


    def run_plots(self, ctx, plots: list[str]):
        for plot_name in plots:
            plot_is_not_skipped = (plot_name, ctx.mode) not in self._skipped_plots
            if plot_name not in PLOT_REGISTRY:
                if plot_is_not_skipped:
                    logger_inst.warning(f"Plot '{plot_name}' is not registered — skipping.")
                    self._skipped_plots.add((plot_name, ctx.mode))
                continue

            spec = PLOT_REGISTRY[plot_name]
            try:
                fig, ax = run_plot(plot_name, ctx)
                if self.tensorboard:
                    self.tensorboard.log_figure(
                        tag=f"{ctx.mode}/{plot_name}",
                        figure=fig,
                        step=ctx.global_step,
                    )
            except RequirementNotMet as e:
                if spec.optional:
                    if plot_is_not_skipped:
                        logger_inst.debug(f"Skipping optional plot '{plot_name} in mode{ctx.mode} ': {e}")
                        self._skipped_plots.add((plot_name, ctx.mode))
                else:
                    # required plot whose requirements aren't met — this IS a bug
                    raise ValueError(
                        f"Required plot '{plot_name}' could not run because: {e}\n"
                        f"Mark it optional=True in register_plot if this is expected."
                    ) from e


    def run_scalars(self, ctx, artifact_names):

        for full_name, artifact_name in artifact_names.items():
            # separate full name in to tag and name of the scalar
            tag, name = full_name.split("/")
            value = run_scalar(ctx=ctx, name=artifact_name)
            self.tensorboard.log_scalar(
                name=tag,
                values={name: value},
                step=ctx.global_step,
            )

class RequirementNotMet(Exception):
    """
    Raised by ensure() when a required artifact cannot be provided
    because the underlying data doesn't exist in this context.
    Caught by run_plots to skip optional plots gracefully.
    """
    def __init__(self, artifact, plot_name=None):
        self.artifact = artifact
        self.plot_name = plot_name
        super().__init__(
            f"Requirement '{artifact}' cannot be met for plot '{plot_name}' "
            f"— no builder provides it and it is not in the context. Skipping."
        )
