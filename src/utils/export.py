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
    add_softmax: bool =True) -> str:
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

    if add_softmax:
        model_inst = create_model.AddActFnToModel(model_inst, "softmax")

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


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Export model to torch-export (.pt2) with dynamic batch dim")
    p.add_argument("--model_path", "-m", required=True, help="Path to the model file (file or checkpoint) to load")
    p.add_argument("--fold", "-f", required=True, help="Fold number", default=0)

    args = p.parse_args()

    model_path = pathlib.Path(args.model_path)
    fold = args.fold
    # if given path is a only a name search for name in environment dir
    if len(model_path.parts) == 1:
        model_path = pathlib.Path(os.environ["MODELS_DIR"]).with_stem(args.model_path).with_suffix(".pt2")

    model = torch.load(args.model_path, weights_only=False)
    # model is dict with keys
    # epoch ,model_inst, model_state_dict, optimizer
    model = model["model_inst"]
    model.eval()
    torch_export(model, name=str(model_path), fold=fold)
    print("Done exporting")
