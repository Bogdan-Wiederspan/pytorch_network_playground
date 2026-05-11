from __future__ import annotations

import pathlib
import argparse
import os

import torch

from models import create_model


def torch_export(
    model_inst: torch.nn.Module,
    name: str,
    fold: str,
    base_dir: str | None=None,
    activation_fn_name: str =None) -> str:
    """
    Takes *model*

    Args:
        model_inst (torch.nn.Module): PyTorch model instance, no path.
        name (str): Name of the model, used to name the result.
        fold (str): Current fold of the model. Is part of the naming scheme
        base_dir (str | None, optional): Directory where model is stored, if None take from MODELS_DIR environment. Defaults to None.
        add_softmax (bool, optional): Adds Softmax as last layer. Defaults to True.

    Returns:
        str: Path to exported pt2 model.
    """
    # by default set CPU device, to enable most compatible export.
    DEVICE=torch.device("cpu")

    if activation_fn_name is not None:
        model_inst = create_model.utils.AddActFnToModel(model_inst, activation_fn_name)

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

    dim = torch.export.Dim("batch")
    dynamic_shapes = {
        "categorical_inputs": {0:dim, 1:categorical_input.shape[-1]},
        "continuous_inputs" : {0:dim, 1:continuous_inputs.shape[-1]},
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
    torch.export.save(exp, dst, pickle_protocol=4)
    return dst


def torch_save(
    model: torch.nn.Module,
    name: str,
    fold: str,
    base_dir: str | None) -> None:
    """
    Small wrapper to save *model_inst* with *name* and *fold* number in *base_dir*.
    If *base_dir* is None a default location defined in MODELS_DIR environment is used.

    Args:
        model (torch.nn.Module): Pytorch model instance.
        name (str): Name of the saved model.
        fold (str): Fold number of the saved model.
        base_dir (str | None): Path to directory where model is saved.
    """
    base_dir = os.environ["MODELS_DIR"]
    dst = (pathlib.Path(base_dir) / f"{name}_fold{fold}").with_suffix(".pt")
    torch.save(model, dst)


def run_exported_tensor_model(
    pt2_path: str,
    cat: torch.tensor,
    cont: torch.tensor) -> torch.tensor:
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

def resolve_models_path(path):
    models_path = pathlib.Path(path)
    is_only_model_name = len(path.parts) == 1
    if is_only_model_name:
        models_root = pathlib.Path(os.environ["MODELS_DIR"])
        models_path = models_root / models_path.stem
    return models_path.with_suffix(".pt2"), models_path.with_suffix(".pt")

def build_model(path):
    checkpoint = torch.load(str(path), weights_only=False, map_location="cpu")
    # when instance is saved load this, otherwise rebuild model from module and class name and load state dict
    if "model_inst" in checkpoint:
        model_inst = checkpoint["model_inst"]
    else:
        full_cfg = checkpoint["full_config"]

        # model_choice = full_cfg.training_config.model_choice
        model_choice = "binned_lbn_dense"
        model_cls = create_model.MODEL_REGISTRY[model_choice]
        model_inst = model_cls(full_cfg)
        model_inst.load_state_dict(checkpoint["model_state_dict"])
    model_inst.eval()
    return model_inst

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Export model to torch-export (.pt2) with dynamic batch dim")
    p.add_argument("--checkpoint_path", "-m", required=True, help="Path to checkpoint file to load")
    p.add_argument("--fold", "-f", required=True, help="Fold number", default=0)
    p.add_argument(
        "--add_activation",
        required=False,
        help="If value is given, get activation function and add at end of network",
        default=None,
        choices=["sigmoid", "softmax"]
        )
    args = p.parse_args()

    pt2_path, pt_path = resolve_models_path(pathlib.Path(args.checkpoint_path))
    fold = args.fold

    model_inst = build_model(pt_path)
    # model is dict with keys
    # epoch ,model_inst, model_state_dict, optimizer
    # we want an model_instance OR if not existing we create new model instance and load the state dict
    torch_export(model_inst, name=str(pt2_path), fold=fold)
    print("Done exporting")
