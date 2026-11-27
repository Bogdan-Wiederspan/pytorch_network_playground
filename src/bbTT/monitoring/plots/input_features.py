import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bbTT.monitoring.register import register_plot
from bbTT.utils.utils import EMPTY_FLOAT

# background process type -> (legend label, color)
BACKGROUND_STYLE = {
    "tt": (r"$t\bar{t}$", "tab:blue"),
    "dy": ("DY", "tab:green"),
}

# true kappa lambda of the signal process ids, used for the legend
# also for networks without physics parameters, where the config does not contain this information
SIGNAL_KL = {21114: 0.0, 21101: 1.0, 21120: 2.45, 21121: 5.0}


def histogram_groups(ctx, process_ids: np.ndarray, signal_kl: dict[int, float]) -> list[tuple[np.ndarray, dict]]:
    """
    Event masks and histogram styles: one group per signal process id (kappa lambda) and one per
    background process type (all process ids combined).
    Signal at kappa lambda = 1 (or a single signal process) is solid red, other kappa lambdas are
    dashed shades of orange to dark red. Backgrounds are solid.

    Returns:
        list[tuple[np.ndarray, dict]]: (event mask, keyword arguments for ax.hist) per group.
    """
    groups = []
    signal_mask = (ctx.targets[:, ctx.target_map["hh"]] == 1).numpy()
    signal_pids = sorted(np.unique(process_ids[signal_mask]), key=lambda pid: signal_kl.get(int(pid), float("inf")))

    def is_reference(pid):
        return len(signal_pids) == 1 or np.isclose(signal_kl.get(int(pid), np.nan), 1.0)

    other_pids = [pid for pid in signal_pids if not is_reference(pid)]
    shades = dict(zip(other_pids, plt.cm.YlOrRd(np.linspace(0.4, 0.95, len(other_pids)))))
    for pid in signal_pids:
        kl = signal_kl.get(int(pid))
        label = f"HH ($\\kappa_\\lambda$={kl:g})" if kl is not None else f"HH (pid {pid})"
        style = {"color": "tab:red", "linestyle": "-"} if is_reference(pid) else {"color": shades[pid], "linestyle": "--"}
        groups.append((signal_mask & (process_ids == pid), {"label": label, **style}))

    for process_type, idx in ctx.target_map.items():
        if process_type == "hh":
            continue
        label, color = BACKGROUND_STYLE.get(process_type, (process_type, None))
        mask = (ctx.targets[:, idx] == 1).numpy()
        groups.append((mask, {"label": label, "color": color, "linestyle": "-"}))
    return groups


@register_plot(
    "input_features",
    requires=("batch_inputs",),
    optional=True,
)
def plot_input_features(ctx, n_bins=30, n_cols=8, **kwargs) -> tuple[Figure, Axes]:
    """
    Histograms of every network input, exactly as they appear in one batch.
    Signal is shown per process id (kappa lambda), backgrounds per process type (tt, dy).

    Values are the raw inputs delivered by the sampler, before the preprocessing inside the model
    (e.g. standardization). Histograms are unweighted event counts: the batch composition of the
    sampler already represents the weighting. Missing values are excluded from the histogram,
    their fraction is shown in the title of the subplot. A value is missing when it equals the
    marker of its feature (``DataConfig.dummy_values`` via ``markers_by_position``, e.g. 0 for missing jets,
    -1 for missing b-tags); inputs without a marker fall back to EMPTY_FLOAT.

    Args:
        ctx (EvalContext): Must carry ``batch_inputs``, a dict with "continuous", "categorical",
            optional "physics_parameters" tensors, their feature names under "names", the process id
            per event under "process_ids", optionally the kappa lambda per signal process id
            under "signal_kl" and optionally the missing-value marker per feature under
            "missing_markers" (same keys as "names").
        n_bins (int): Number of bins for continuous inputs.
        n_cols (int): Number of subplots per row.
        **kwargs: Unused, present for registry call-signature consistency.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The figure and axes.
    """
    batch_inputs = ctx.get("batch_inputs")
    # values from the config take precedence over the default mapping
    signal_kl = {**SIGNAL_KL, **batch_inputs.get("signal_kl", {})}
    groups = histogram_groups(ctx, batch_inputs["process_ids"].cpu().numpy(), signal_kl)

    # (name, values, is_categorical, missing-value marker) for every input column
    columns = []
    missing_markers = batch_inputs.get("missing_markers", {})
    for kind, is_categorical in (("continuous", False), ("categorical", True), ("physics_parameters", True)):
        tensor = batch_inputs.get(kind)
        if tensor is None:
            continue
        values = tensor.detach().cpu().to(torch.float64).numpy()
        names = batch_inputs["names"][kind]
        markers = missing_markers.get(kind) or [EMPTY_FLOAT] * len(names)
        for idx, name in enumerate(names):
            # markers are compared in float32, the precision the inputs are stored in
            columns.append((name, values[:, idx], is_categorical, float(np.float32(markers[idx]))))

    n_rows = math.ceil(len(columns) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (name, values, is_categorical, marker) in zip(axes, columns):
        present = values != marker
        if not present.any():
            ax.set_title(f"{name}  (all missing)", fontsize=9)
            continue

        if is_categorical:
            # one evenly spaced bin per value that exists in the batch, labeled with the value
            # (also for non integer values like kappa lambda)
            unique = np.unique(values[present])
            values = np.where(present, np.searchsorted(unique, values), values)
            edges = np.arange(len(unique) + 1) - 0.5
            ax.set_xticks(np.arange(len(unique)))
            ax.set_xticklabels([f"{u:g}" for u in unique])
        else:
            low, high = np.percentile(values[present], [0.5, 99.5])
            edges = np.linspace(low, high, n_bins + 1) if high > low else np.array([low - 0.5, low + 0.5])

        for mask, style in groups:
            selected = values[mask & present]
            ax.hist(
                np.clip(selected, edges[0], edges[-1]),  # overflow into first and last bin
                bins=edges,
                histtype="step",
                linewidth=1.5,
                **style,
            )

        title = name
        missing = 1 - present.mean()
        if missing > 0:
            title += f"  ({missing:.1%} missing)"
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)

    # hide unused subplots
    for ax in axes[len(columns):]:
        ax.set_visible(False)

    # one legend row between title and subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        f"Network inputs of batch at step {ctx.global_step} ({len(ctx.targets)} events, unweighted)",
        fontsize=14,
        y=0.998,
    )
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.982), ncol=len(labels), fontsize=12, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return fig, axes
