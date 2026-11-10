from __future__ import annotations

import copy
import dataclasses
import hashlib
import pathlib
import re
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from bbTT.monitoring.logger.logger import get_logger

if TYPE_CHECKING:
    from bbTT.configs.full_config import FullConfig
    from bbTT.train.setup import TrainingComponents

logger_inst = get_logger(__name__)

# Which sub-config cant be overwritten
# reasoning: if you change a models definition you can't load its model parameter
PROTECTED_OVERRIDES: tuple[str, ...] = ("model_building_config",)

# overrides below this prefix change what the validation loss means, so the best loss of the checkpoint
# is no longer comparable and the checkpoint criteria start fresh
LOSS_OVERRIDE_PREFIX = "loss_config"

RESUME_TAG_SEPARATOR = "-resume-"


@dataclasses.dataclass
class ResumeState:
    """
    Everything a resumed training needs that a fresh training does not.

    Args:
        checkpoint (dict[str, Any]): Loaded checkpoint, see CheckPoint._snapshot for its keys.
        full_config (FullConfig): Checkpoint config with the requested overrides applied.
        fold (int): Fold the checkpoint was trained on, the only fold the resume trains.
        start_iteration (int): First iteration of the resumed loop.
        tag (str): Short hash of the overrides, suffix for checkpoint and tensorboard names of the resume.
        keep_checkpoint_criteria (bool): Whether best loss and counters of the checkpoint still apply.
    """

    checkpoint: dict[str, Any]
    full_config: FullConfig
    fold: int
    start_iteration: int
    tag: str
    keep_checkpoint_criteria: bool

    def run_name(self, base_name: str) -> str:
        """
        Args:
            base_name (str): Name of the original run, e.g. the tensorboard name or the checkpoint name.

        Returns:
            str: Name of the resumed run, derived from the original so both sort next to each other.
        """
        return f"{base_name}{RESUME_TAG_SEPARATOR}{self.tag}"


# --- config merge -------------------------------------------------------------------------------------------
# firs the config from checkpoint is loaded than overwrite happens.

def _split_path(path: str) -> list[str]:
    """
    Args:
        path (str): Dotted path into FullConfig, e.g. "record_config.validation_interval".

    Returns:
        list[str]: Path segments.

    Raises:
        ValueError: On an empty path or empty segment.
    """
    segments = path.split(".")
    if not path or any(not s for s in segments):
        raise ValueError(f"invalid override path '{path}'")
    return segments


def _resolve_parent(root: Any, segments: Sequence[str]) -> Any:
    """
    Walk all but the last segment, so the caller can get or set the last attribute on the returned object.

    Args:
        root (Any): Object to start from.
        segments (Sequence[str]): Path segments, at least one.

    Returns:
        Any: Object holding the attribute named by the last segment.

    Raises:
        AttributeError: If a segment does not exist, the message names the full path.
    """
    obj = root
    for segment in segments[:-1]:
        if not hasattr(obj, segment):
            raise AttributeError(f"override path '{'.'.join(segments)}': '{segment}' not found")
        obj = getattr(obj, segment)
    if not hasattr(obj, segments[-1]):
        raise AttributeError(f"override path '{'.'.join(segments)}': '{segments[-1]}' not found")
    return obj


def _rerun_post_init(config: Any) -> None:
    """
    Run ``__post_init__`` again on *config* and every dataclass below it.

    A config rebuilt from a checkpoint dict skipped ``__post_init__``, so fields derived there (e.g.
    DataConfig.datasets from dataset_pattern and eras) still hold the checkpoint's values after an override.
    Re-running makes the merged config equal to a freshly constructed one with the same fields.
    This should only matter, when an overwrite contains 'wrong' values.

    Args:
        config (Any): Dataclass instance.
    """
    for field in dataclasses.fields(config):
        value = getattr(config, field.name)
        if dataclasses.is_dataclass(value):
            _rerun_post_init(value)
    post_init = getattr(config, "__post_init__", None)
    if post_init is not None:
        post_init()



def merge_config(base: FullConfig, current: FullConfig, overrides: Iterable[str]) -> FullConfig:
    """
    Take *base* and replace the attributes named in *overrides* by their values in *current*.

    Both a whole sub config ("record_config") and a single field ("record_config.validation_interval")
    are valid paths.
    Afterwards every ``__post_init__`` in the config tree runs again, so derived fields
    follow the overrides and the compatibility rules of FullConfig are checked.

    Args:
        base (FullConfig): Config stored in the checkpoint, the source of truth for the model.
        current (FullConfig): Config of the running code, the source of the overridden values.
        overrides (Iterable[str]): Dotted paths to take from *current*.

    Returns:
        FullConfig: Deep copy of *base* with the overrides applied.

    Raises:
        ValueError: If a path starts with a protected sub config.
    """
    merged = copy.deepcopy(base)
    for path in overrides:
        segments = _split_path(path)
        if segments[0] in PROTECTED_OVERRIDES:
            raise ValueError(
                f"override '{path}' is not allowed: {segments[0]} defines the architecture the checkpoint was written for"
            )
        source_parent = _resolve_parent(current, segments)
        target_parent = _resolve_parent(merged, segments)
        value = copy.deepcopy(getattr(source_parent, segments[-1]))
        setattr(target_parent, segments[-1], value)
        logger_inst.i_info(f"Resume override: {path} = {value!r}")

    # from_dict skipped __post_init__: recompute derived fields and re-check the compatibility rules
    _rerun_post_init(merged)
    return merged


def override_tag(current: FullConfig, overrides: Iterable[str]) -> str:
    """
    Short hash over the override paths and their values, so different continue settings get different names.

    Args:
        current (FullConfig): Config the overridden values come from.
        overrides (Iterable[str]): Dotted paths, see merge_config.

    Returns:
        str: 8 hex characters, "base" when there are no overrides.
    """
    items = []
    for path in sorted(overrides):
        segments = _split_path(path)
        value = getattr(_resolve_parent(current, segments), segments[-1])
        if dataclasses.is_dataclass(value):
            value = dataclasses.asdict(value)
        items.append((path, repr(value)))
    if not items:
        return "base"
    return hashlib.sha256(repr(items).encode("utf-8")).hexdigest()[:8]


def loss_changed(overrides: Iterable[str]) -> bool:
    """
    Args:
        overrides (Iterable[str]): Dotted paths, see merge_config.

    Returns:
        bool: True if any override touches the loss config.
    """
    return any(_split_path(p)[0] == LOSS_OVERRIDE_PREFIX for p in overrides)

def override_paths(config: Any, prefix: str = "") -> list[str]:
    """
    All dotted paths that merge_config accepts for *config*: every dataclass field, and every sub config as a whole.

    Args:
        config (Any): Dataclass instance to walk, normally a FullConfig.
        prefix (str): Path prefix for recursion, empty at the top level.

    Returns:
        list[str]: Paths in field order; a sub config is listed before its fields. Protected sub configs are
            listed too, so the caller can mark them.
    """
    paths: list[str] = []
    for field in dataclasses.fields(config):
        path = f"{prefix}{field.name}"
        paths.append(path)
        value = getattr(config, field.name)
        if dataclasses.is_dataclass(value):
            paths.extend(override_paths(value, prefix=f"{path}."))
    return paths

def format_overrides(config: Any) -> str:
    """
    Human readable listing of all override paths with their value in *config*, for --list-overrides.

    Args:
        config (Any): The current code config, a FullConfig.

    Returns:
        str: One line per path, sub configs as headers, protected ones marked.
    """
    lines = ["Override paths (values from the current code config):"]
    for path in override_paths(config):
        segments = path.split(".")
        value = getattr(_resolve_parent(config, segments), segments[-1])
        if dataclasses.is_dataclass(value):
            note = "  [protected, not overridable]" if segments[0] in PROTECTED_OVERRIDES else ""
            lines.append(f"\n{path}{note}")
        else:
            lines.append(f"  {path} = {value!r}")
    return "\n".join(lines)



# --- loading ------------------------------------------------------------------------------------------------


def _fold_of(checkpoint: dict[str, Any], path: pathlib.Path) -> int:
    """
    Args:
        checkpoint (dict[str, Any]): Loaded checkpoint.
        path (pathlib.Path): File it was loaded from.

    Returns:
        int: Fold from the checkpoint, or parsed from the file name "<name>_fold<k>.pt" for older checkpoints.

    Raises:
        ValueError: If neither source yields a fold.
    """
    if "fold" in checkpoint:
        return int(checkpoint["fold"])
    match = re.search(r"_fold(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"cannot determine fold of checkpoint {path}: no 'fold' key and no '_fold<k>' in the name")
    return int(match.group(1))


def load_resume_state(
    checkpoint_name: str | pathlib.Path,
    current_config: FullConfig,
    overrides: Sequence[str] = (),
) -> ResumeState:
    """
    Load a checkpoint and prepare the config of the resumed training.

    The checkpoint config is the base; the overrides are taken from *current_config*. Unless
    `training_config.save_model_name` is overridden, the resume writes its own checkpoints under
    "<name>-resume-<tag>", so the loaded checkpoint is never overwritten and stays an anchor.

    Args:
        checkpoint_name (str | pathlib.Path): Checkpoint name or path, see loading.resolve_checkpoint_path.
        current_config (FullConfig): Config of the running code.
        overrides (Sequence[str]): Dotted paths to take from *current_config*, see merge_config.

    Returns:
        ResumeState: Checkpoint, merged config, fold, start iteration and naming tag.
    """
    from bbTT.checkpoint.loading import load_checkpoint, resolve_checkpoint_path
    from bbTT.configs.full_config import FullConfig
    from bbTT.evaluation.rebuild_dataclasses import from_dict

    path = resolve_checkpoint_path(checkpoint_name, suffix=".pt")
    checkpoint = load_checkpoint(path)
    base_config = from_dict(FullConfig, checkpoint["full_config"])

    merged = merge_config(base=base_config, current=current_config, overrides=overrides)
    tag = override_tag(current_config, overrides)
    if "training_config.save_model_name" not in overrides:
        merged.training_config.save_model_name = f"{base_config.training_config.save_model_name}{RESUME_TAG_SEPARATOR}{tag}"

    keep_criteria = not loss_changed(overrides)
    if not keep_criteria:
        logger_inst.i_info("Resume with changed loss: checkpoint criteria (best loss, patience) start fresh")

    fold = _fold_of(checkpoint, path)
    logger_inst.i_info(
        f"Resume from {path.name}: fold {fold}, iteration {checkpoint['iteration']}, "
        f"new checkpoints under '{merged.training_config.save_model_name}'"
    )
    return ResumeState(
        checkpoint=checkpoint,
        full_config=merged,
        fold=fold,
        start_iteration=int(checkpoint["iteration"]) + 1,
        tag=tag,
        keep_checkpoint_criteria=keep_criteria,
    )


def restore_training_state(components: TrainingComponents, resume_state: ResumeState) -> int:
    """
    Load the checkpoint into freshly built components.

    Model, optimizer and scheduler get their state dicts. The CheckPoint gets the loaded checkpoint as
    `last_checkpoint`, so the scheduler handler can roll back to it after a learning rate drop, and its
    criteria (best loss, patience counter) when they still apply.

    Args:
        components (TrainingComponents): Components built from the merged config, see setup.build_training_components.
        resume_state (ResumeState): What load_resume_state returned.

    Returns:
        int: Iteration the loop starts with.
    """
    checkpoint = resume_state.checkpoint
    components.model_inst.load_state_dict(checkpoint["model_state_dict"])
    components.optimizer_inst.load_state_dict(checkpoint["optimizer_state_dict"])
    if components.scheduler_inst is not None and checkpoint.get("scheduler_state_dict") is not None:
        components.scheduler_inst.load_state_dict(checkpoint["scheduler_state_dict"])

    checkpoint_inst = components.checkpoint_inst
    checkpoint_inst.last_checkpoint = checkpoint
    if resume_state.keep_checkpoint_criteria and "checkpoint_state_dict" in checkpoint:
        checkpoint_inst.load_state_dict(checkpoint["checkpoint_state_dict"])

    return resume_state.start_iteration
