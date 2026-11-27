from __future__ import annotations

import os
import pathlib
import sys

import torch

from bbTT.checkpoint.loading import rebuild_checkpoint_information
from bbTT.models.utils.append_layers import ACTIVATION_CHOICE, HEAD_CHOICE, AddActFnToModel, SelectHead
from bbTT.monitoring.logger.logger import get_logger
from bbTT.utils.parser import ParserBuilder

logger_inst = get_logger(__name__)


def export_pt2(
    model_inst: torch.nn.Module,
    name: str,
    fold: str,
    base_dir: str | None = None,
    head: HEAD_CHOICE = "probabilities",
    add_activation: ACTIVATION_CHOICE = "identity",
) -> pathlib.Path:
    """
    Takes *model_inst* and export it to pt2 format, with dynamic batch dimension.
    The exported model is stored as ``<base_dir>/<name>_fold<fold>.pt2``.

    The model's forward returns logits; the exported graph returns the selected *head* (see BaseModel.HEADS).
    An extra activation can be wrapped on top of that head (*add_activation*), which only makes sense on the
    "logits" head. Stacking it on an activated head is allowed, but logged as warning since it is rarely intended.

    Args:
        model_inst (torch.nn.Module): PyTorch model instance, no path.
        name (str): Name of the model, used to name the result.
        fold (str): Current fold of the model. Is part of the naming scheme
        base_dir (str | None, optional): Directory where model is stored, if None take from MODELS_DIR environment. Defaults to None.
        head (HEAD_CHOICE, optional): Head the exported model returns. Defaults to "probabilities".
        add_activation (ACTIVATION_CHOICE, optional): Extra activation applied on top of the selected head, "identity" adds nothing.
            Defaults to "identity".
    Returns:
        pathlib.Path: Path to exported pt2 model.
    """
    # by default set CPU device, to enable most compatible export.
    DEVICE = torch.device("cpu")

    if head != "logits" and add_activation != "identity":
        logger_inst.warning(
            f"Head {head!r} is already activated, adding another activation on top: {add_activation}. "
            "This may lead to unexpected results."
            )

    model_inst = SelectHead(model_inst, head)

    if add_activation != "identity":
        model_inst = AddActFnToModel(model_inst, add_activation)

    # prepare model_inst
    model_inst = model_inst.to(DEVICE)
    model_inst = model_inst.eval()

    # prepare model input signature, which requires the exact shape - using dummy inputs
    num_cat = len(model_inst.categorical_features)
    num_cont = len(model_inst.continuous_features)

    # HINT: batch size 1 is an edge case and overwrites dynamic batch size
    # batch size is set to 2 instead to prevent this.
    categorical_input = torch.zeros(2, num_cat).to(torch.int32).to(DEVICE)
    continuous_inputs = torch.zeros(2, num_cont).to(torch.float32).to(DEVICE)

    # specify the ranges of dimension so that the safeguard for export is statist
    dim = torch.export.Dim("batch", min = 2, max = 1_000_000)
    dynamic_shapes = {
        "categorical_inputs": {0: dim, 1: categorical_input.shape[-1]},
        "continuous_inputs": {0: dim, 1: continuous_inputs.shape[-1]},
    }

    # do actual export and saving
    exp = torch.export.export(
        model_inst,
        args=(categorical_input, continuous_inputs),
        dynamic_shapes=dynamic_shapes,
    )

    if base_dir is None:
        base_dir = os.environ["MODELS_DIR"]

    dst = (pathlib.Path(base_dir) / f"{name}_fold{fold}").with_suffix(".pt2")
    torch.export.save(exp,dst)
    return dst


def run_pt2(pt2_path: str, cat: torch.tensor, cont: torch.tensor) -> torch.tensor:
    """
    Run a given model stored at *pt2_path* with *cat* and *cont* tensors.
    This is a method to check if exporting the model resulted in same results.

    Args:
        pt2_path (str): Path to Pt2 file.
        cat (torch.tensor): Tensor with categorical features.
        cont (torch.tensor): Tensor with continuous features.

    Returns:
        torch.tensor: Tensor with model output.
    """
    exp = torch.export.load(pt2_path)
    scores = exp.module()(cat, cont)
    return scores


def compare_pt2_with_original(
    pt2_path: str, original_model: torch.nn.Module, cat: torch.tensor, cont: torch.tensor, atol: float = 1e-6
) -> bool:
    """
    Compare the output of a pt2 model with the output of the original model for given *cat* and *cont* tensors. Returns True if outputs are close within given *atol*, False otherwise.
    """
    pt2_output = run_pt2(pt2_path, cat, cont)
    original_output = original_model(cat, cont)
    return torch.allclose(pt2_output, original_output, atol=atol)


def resolve_models_path(path: pathlib.Path, folds) -> tuple[pathlib.Path, pathlib.Path]:
    """
    Small helper to resolve *path* to the model, if only model name is given, look in MODELS_DIR for it, otherwise take given path
    Returns a tuple of the resolved pt2 path and the original pt path, which is needed to load the original model instance for comparison.
    """
    models_path = pathlib.Path(path)
    is_only_model_name = len(path.parts) == 1
    paths = {}
    for fold in folds:
        if is_only_model_name:
            models_root = pathlib.Path(os.environ["MODELS_DIR"])
            models_path = models_root / f"{models_path.stem}_fold{fold}"
        paths[fold] = models_path
    return paths


def main(argv: list[str] | None = None) -> int:
    """
    Command line entry point (``bbtt-export``): export the checkpoints of the given folds to .pt2.

    Args:
        argv (list[str] | None): Arguments without the program name, ``None`` reads ``sys.argv``;
            an empty list prints the help and does nothing (convention of the bbtt-* tools).

    Returns:
        int: Exit code, 0 on success.
    """
    if argv is None:
        argv = sys.argv[1:]
    parser = ParserBuilder(
        "load_checkpoint",
        "head",
        "activation_fn",
        description="Export a model checkpoint to torch-export (.pt2) with a dynamic batch dimension.",
        epilog="example:\n  bbtt-export --checkpoint_src my_first_run --fold 0",
        argv=argv or ["--help"],
    )
    args = parser.args

    paths = resolve_models_path(args.path, args.fold)

    # resolved paths already carry the fold suffix, the export adds it again based on the plain model name
    for fold, path in paths.items():
        logger_inst.info(f"Export fold {fold} with model checkpoint from {path}")
        model_inst, full_cfg = rebuild_checkpoint_information(path.with_suffix(".pt"))

        dst = export_pt2(
            model_inst,
            name=args.path.stem,
            fold=fold,
            head=args.head,
            add_activation=args.add_activation,
        )
    logger_inst.info(f"Finished exporting models for folds {args.fold} with name {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
