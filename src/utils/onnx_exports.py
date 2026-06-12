from __future__ import annotations

import numpy as np
import torch

from utils.utils import maybe_import

onnx = maybe_import("onnx")
rt = maybe_import("onnxruntime")


def export_ensemble_onnx(
    ensemble_wrapper,
    categorical_tensor: torch.Tensor,
    continuous_tensor: torch.Tensor,
    save_dir: str,
    opset_version: int = None,
) -> str:
    """
    Wrapper to export an ensemble model to onnx format. Does the same as export_onnx, but
    iterates over all models in the ensemble_wrapper and freezes them before exporting.

    Args:
        ensemble_wrapper (MLEnsembleWrapper): _description_
        categorical_tensor (torch.tensor): tensor representing categorical features
        continuous_tensor (torch.tensor): tensor representing categorical features
        save_dir (str): directory where the onnx model will be saved.
        opset_version (int, optional): version of the used operation sets. Defaults to None.

    Returns:
        str: The path of the saved onnx ensemble model.
    """
    for model in ensemble_wrapper.models:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    export_onnx(
        ensemble_wrapper,
        categorical_tensor,
        continuous_tensor,
        save_dir,
        opset_version=opset_version,
    )


def export_onnx(
    model: torch.nn.Module,
    categorical_tensor: torch.Tensor,
    continuous_tensor: torch.Tensor,
    save_dir: str,
    opset_version: int = None,
) -> str:
    """
    Function to export a loaded pytorch *model* to onnx format saved in *save_dir*.
    To successfully export the model, the input tensors *categorical_tensor* and *continuous_tensor* must be provided.
    For backwards compatibility, an opset_version can be enforced.
    A table about which opsets are available can be found here: https://onnxruntime.ai/docs/reference/compatibility.html
    Some operations are only available in newer opsets, or change behavior inbetween version.
    A list of all operations and their versions is given at: https://onnx.ai/onnx/operators/

    Args:
        model (torch.nn.model): loaded Pytorch model, ready to perform inference.
        categorical_tensor (torch.tensor): tensor representing categorical features
        continuous_tensor (torch.tensor): tensor representing categorical features
        save_dir (str): directory where the onnx model will be saved.
        opset_version (int, optional): version of the used operation sets. Defaults to None.

    Returns:
        str: The path of the saved onnx model.
    """

    # logger = law.logger.get_logger(__name__)

    onnx_version = onnx.__version__
    runtime_version = rt.__version__
    torch_version = torch.__version__

    save_path = f"{save_dir}-onnx_{onnx_version}-rt_{runtime_version}-torch{torch_version}.onnx"

    # prepare export
    num_cat_features = categorical_tensor.shape[-1]
    num_cont_features = continuous_tensor.shape[-1]

    # cast to proper format, numpy and float32
    categorical_tensor = categorical_tensor.numpy().astype(np.float32).reshape(-1, num_cat_features)
    continuous_tensor = continuous_tensor.numpy().astype(np.float32).reshape(-1, num_cont_features)

    # double bracket is necessary since onnx, and our model unpacks the input tuple
    input_feed = ((categorical_tensor, continuous_tensor),)

    torch.onnx.export(
        model,
        input_feed,
        save_path,
        input_names=["cat", "cont"],
        output_names=["output"],
        # if opset is none highest available will be used

        opset_version=opset_version,
        do_constant_folding=True,
        dynamic_axes={
            # enable dynamic batch sizes
            "cat": {0: "batch_size"},
            "cont": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    # logger.info(f"Succefully exported onnx model to {save_path}")
    return save_path


def test_run_onnx(
    model_path: str,
    categorical_array: np.ndarray,
    continuous_array: np.ndarray,
) -> np.ndarray:
    """
    Function to run a test inference on a given *model_path*.
    The *categorical_array* and *continuous_array* are expected to be given as numpy arrays.

    Args:
        model_path (str): Model path to onnx model
        categorical_array (np.ndarray): Array of categorical features
        continuous_array (np.ndarray): Array of continuous features

    Returns:
        np.ndarray: Prediction of the model
    """
    sess = rt.InferenceSession(model_path, providers=rt.get_available_providers())
    first_node = sess.get_inputs()[0]
    second_node = sess.get_inputs()[1]

    # setup data
    input_feed = {
        first_node.name: categorical_array.reshape(-1, first_node.shape).astype(np.float32),
        second_node.name: continuous_array.reshape(-1, second_node.shape).astype(np.float32),
    }

    output_name = [output.name for output in sess.get_outputs()]

    onnx_predition = sess.run(output_name, input_feed)
    return onnx_predition
