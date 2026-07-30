from .register import BUILDER_REGISTRY, PLOT_REGISTRY

_PROVIDER_MAP_CACHE = None

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
        required_plots = require_map(PLOT_REGISTRY).get(artifact, [])
        required_builder = require_map(BUILDER_REGISTRY).get(artifact, [])
        msg = (
            f"Requester: {requester} require {artifact} \n"
            f" but no builder provides'\n"
            "Following plots / builder require this artifact\n"
            f"Plots: {required_plots}\n"
            f"Builders:{required_builder}\n"
            f"Active providers are: {providers}"
        )
        raise ValueError(msg)

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

    def run_plots(
        self,
        ctx,
        plots: list[str],
    ):
        for plot_name in plots:
            fig, ax = run_plot(plot_name, ctx)
            if self.tensorboard:
                self.tensorboard.log_figure(
                    tag=f"{ctx.mode}/{plot_name}",
                    figure=fig,
                    step=ctx.global_step,
                )

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
