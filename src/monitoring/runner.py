from .register import BUILDER_REGISTRY, PLOT_REGISTRY

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

def build_provider_map():
    # get mapping from provides -> provider
    providers = {}

    for name, spec in BUILDER_REGISTRY.items():
        for artifact in spec.provides:
            providers[artifact] = name

    return providers


def ensure(ctx, artifact, providers):
    # check if artifact is already provided otherwise build it via builder
    # for builder ensure that their dependencies also exist
    if ctx.has(artifact):
        return

    builder_name = providers.get(artifact)

    if builder_name is None:
        required_plots = require_map(PLOT_REGISTRY)[artifact]
        required_builder = require_map(BUILDER_REGISTRY)[artifact]
        msg = (
            f"No builder provides '{artifact}'\n"
            "Following plots / builder require this artifact\n"
            f"Plots: {required_plots}\n"
            f"Builders:{required_builder}\n"
            f"Active providers are: {providers}"
        )
        raise ValueError(msg)

    builder = BUILDER_REGISTRY[builder_name]

    for dep in builder.requires:
        ensure(ctx, dep, providers)

    result = builder.fn(ctx)
    # save result in cache of context
    ctx.cache.update(result)

def run_plot(name, ctx):

    spec = PLOT_REGISTRY[name]

    providers = build_provider_map()

    for req in spec.requires:
        ensure(ctx, req, providers)

    return spec.fn(ctx)


def run_scalar(name, ctx):
    # scalars
    providers = build_provider_map()
    ensure(ctx, name, providers)
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
