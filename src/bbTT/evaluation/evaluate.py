from __future__ import annotations

# standard imports
from typing import Iterable, Literal, Sequence

# package imports
import numpy as np
import torch

# personal imports
from bbTT.data_handling import io
from bbTT.data_handling.preprocessing import k_fold
from bbTT.evaluation.load_models import rebuild_checkpoint_information
from bbTT.monitoring.logger.logger import get_logger
from bbTT.utils.parser import ParserBuilder

logger_inst = get_logger(__name__)


def last_fn_picker(last_fn):
    def sigmoid(x):
        return torch.sigmoid(x)

    def softmax(x):
        return torch.softmax(x, dim=-1)

    def identity(x):
        return x

    choices = {"sigmoid": sigmoid, "softmax": softmax, "none": identity}
    return choices[last_fn.lower()]


def physics_parameter_hypotheses(
    dataset_config: Literal["DataClass"],
    values: Sequence[float] | None = None,
) -> list[tuple[float, ...]] | None:
    """
    Hypotheses of the physics parameters (e.g. kappa lambda) a network is evaluated at.
    If *values* is None, the values the network was trained on are used.

    Args:
        dataset_config (Literal["DataClass"]): DataConfig of the network, describes the physics parameters.
        values (Sequence[float] | None, optional): Explicit hypotheses, only possible for a single physics parameter.

    Returns:
        list[tuple[float, ...]] | None: Sorted hypotheses or None if the network has no physics parameters.

    Raises:
        ValueError: If *values* are given for a network without exactly one physics parameter.
    """
    if not dataset_config.has_physics_parameters:
        if values:
            raise ValueError("Physics parameter values are given, but the network has no physics parameters")
        return None

    if values:
        if dataset_config.num_physics_parameters != 1:
            raise ValueError("Explicit physics parameter values are only supported for a single physics parameter")
        return sorted((float(v),) for v in set(values))

    # every process id that has values for all physics parameters is a training point
    pids = set.intersection(*(set(v) for v in dataset_config.physics_parameters.values()))
    return sorted({dataset_config.physics_parameter_values(pid) for pid in pids})


def hypothesis_key(hypothesis: tuple[float, ...]) -> float | tuple[float, ...]:
    # single physics parameter: use the plain value as key, e.g. scores[1.0]
    return hypothesis[0] if len(hypothesis) == 1 else hypothesis


def evaluate_model_on_fold(
    events: dict[torch.Tensor],
    model_inst: torch.nn.Module,
    full_config: Literal["DataClass"],
    folds: Iterable[int],
    evaluate_on: str,
    last_activation_fn: Literal["sigmoid", "softmax", "identity"] = "identity",
    batch_size=1,
    physics_parameter_values: Sequence[float] | None = None,
) -> torch.tensor:
    """
    Function to evaluate *folds* with given *model_inst*.
    The configuration if handled by *full_config* and describes the network when it was trained.
    By default only the test set it evaluated, but *evaluate_on* can be extended to also evaluate training and validation set again.

    If the network has physics parameters (e.g. kappa lambda), every event is evaluated once per hypothesis,
    where all events (signal and background) get the same value. "scores" is then a dict: hypothesis -> scores.

    Args:
        model_inst (torch.nn.Module): Loaded model instance, that is used to evaluate the data.
        full_config (Literal[&quot;DataClass&quot;]): Dataclass containing all other configs, is also located in the checkpoint and describes the network when it was trained.
        folds (Iterable[int]): Iterable of folds to evaluate on. Careful: Network should match the fold it was trained on, no check is performed.
        evaluate_on (Iterable[str]): Iterable containing: "test", "training" or "validation", defines which indices are used of the input data used for training.
        physics_parameter_values (Sequence[float], optional): Hypotheses of the physics parameter, only for networks
            with a single physics parameter. Defaults to None, which uses the values the network was trained on.

    Raises:
        ValueError: When wrong value in evaluate_on is used.

    Returns:
        torch.Tensor: Tensor with output scores of the network.
    """
    with torch.no_grad():
        model_inst.eval()

        # --- load data and split by indices ---
        columns_to_split = (
            "event_id",
            "normalization_weights",
            "product_of_weights",
            "evaluation_mask",
        )

        num_targets = len(full_config.dataset_config.target_map.values())
        last_fn = last_fn_picker(last_activation_fn)
        hypotheses = physics_parameter_hypotheses(full_config.dataset_config, physics_parameter_values)
        if hypotheses is not None:
            logger_inst.info(
                f"Evaluate physics parameters {full_config.dataset_config.physics_parameter_names} "
                f"at hypotheses: {[hypothesis_key(h) for h in hypotheses]}"
            )

        dnn_scores = {}
        for fold in folds:
            fold_split_coordinator = k_fold.FoldAndSplitCoordinator(
                events=events,
                c_fold=fold,
                k_fold=full_config.training_config.k_fold,
                seed=full_config.training_config.seed,
                training_percentage=full_config.training_config.train_ratio,
                randomize=full_config.training_config.randomize
            )
            # --- run model on specific splits ---
            dnn_scores[fold] = {}
            logger_inst.info(f"Evaluate {evaluate_on} data of fold {fold}")

            for uid in events.keys():
                idx = fold_split_coordinator.indices[uid][evaluate_on]

                # when no events left after split move on
                total_n = len(idx)
                if total_n == 0:
                    continue

                def score_events(hypothesis: tuple[float, ...] | None) -> torch.Tensor:
                    # all events are evaluated with the same physics parameter values *hypothesis*
                    out_shape = (total_n, num_targets)
                    uid_scores = torch.empty(out_shape, dtype=torch.float32)

                    for start_idx in range(0, total_n, batch_size):
                        end_idx = min(start_idx + batch_size, total_n)
                        batch_idx = idx[start_idx:end_idx]

                        continuous_inputs = events[uid]["continuous"][batch_idx]
                        categorical_inputs = events[uid]["categorical"][batch_idx]
                        physics_parameters = None
                        if hypothesis is not None:
                            physics_parameters = torch.tensor(hypothesis, dtype=torch.float32).expand(len(batch_idx), -1)

                        scores = model_inst(
                            categorical_inputs=categorical_inputs,
                            continuous_inputs=continuous_inputs,
                            physics_parameters=physics_parameters,
                        )

                        # in multi-head networks the first output is always the normal prediction
                        if model_inst.is_binned:
                            scores = scores[0]

                        scores = last_fn(scores)
                        uid_scores[start_idx:end_idx] = scores.detach().cpu()
                    return uid_scores

                # without physics parameters "scores" is a tensor, otherwise a dict: hypothesis -> tensor
                if hypotheses is None:
                    uid_scores = score_events(None)
                else:
                    uid_scores = {hypothesis_key(h): score_events(h) for h in hypotheses}

                dnn_scores[fold][uid] = {
                    "scores": uid_scores,
                    "fold_index": idx,
                }

                # add splitted content on top
                for column in columns_to_split:
                    dnn_scores[fold][uid][column] = events[uid][column][idx]

            del uid_scores
        return dnn_scores


if __name__ == "__main__":
    parser = ParserBuilder(
        "load_checkpoint",
        "activation_fn",
        "save_path",
        "evaluate_choices",
        "batching",
        "num_threading",
        "physics_parameter_values",
        description="Evaluate a model checkpoint on the test, training or validation set of its input data.",
        epilog=(
            "example:\n"
            '    python src/bbTT/evaluation/evaluate.py --c_src "${MODELS_DIR}/my_first_run_fold0.pt" --fold 0 -bs 4096 -fp my_first_run'
        ),
    )
    args = parser.args
    model_name = args.path.stem

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger_inst.info(f"DEVICE: {DEVICE}")

    model_inst, full_config = rebuild_checkpoint_information(args.path)
    # is part of the name so everybody nows which data was used to train the network
    used_data_hash = full_config.dataset_config.content_hash()
    torch.manual_seed(full_config.training_config.seed)
    np.random.seed(full_config.training_config.seed)

    events = io.get_data(full_config.dataset_config, rebuild_cache=False, save_cache=False)
    # evaluate data on each part separate and then flush scores
    for evaluate_on in args.evaluate_on:
        stem = args.file_path.stem
        suffix = args.file_path.suffix

        path = args.file_path.with_stem(f"{used_data_hash}_{model_name}_{evaluate_on}_{stem}").with_suffix(suffix)

        evaluated_data = evaluate_model_on_fold(
            model_inst=model_inst,
            full_config=full_config,
            folds=args.fold,
            evaluate_on=evaluate_on,
            last_activation_fn=args.add_activation,
            events=events,
            batch_size=args.batch_size,
            physics_parameter_values=args.physics_parameter_values,
        )

        torch.save(evaluated_data, path)
