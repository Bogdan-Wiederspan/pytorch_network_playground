import numpy as np
import pandas as pd


def to_numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy()

def compute_bin_edges(n_intervals: int, use_logit: bool, eps: float = 1e-3) -> np.ndarray:
    if use_logit:
        logit = lambda p: np.log(p / (1 - p))
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        edges = sigmoid(np.linspace(logit(eps), logit(1 - eps), n_intervals + 1))
        edges[0], edges[-1] = 0.0, 1.0  # edges are by hand to 0 and 1
    else:
        edges = np.linspace(0.0, 1.0, n_intervals + 1)
    return edges

def percentile_range(values: np.ndarray, low: float = 1.0, high: float = 99.0) -> tuple[float, float]:
    return float(np.percentile(values, low)), float(np.percentile(values, high))


def build_feature_frame(
    data: dict,
    score_data: dict,
    continuous_names: list[str],
    categorical_names: list[str]
    ) -> pd.DataFrame:
    """
    Flatten the dict into one Pandas DataFrame with named feature columns + a label column.
    The label columns is taken from names and should represent the order in the tensor.

    Args:
    data (dict): Expect Torch Tensors to be of form: {pid: {"continuous": torch.Tensor, "categorical": torch.Tensor}}
    continuous_names (list[str]):  Column names for the continuous tensor, in order.
    categorical_names (list[str]): Column names for the categorical tensor, in order.

    Returns:
        (pd.DataFrame): One row per event, columns = continuous_names + categorical_names + "label".
    """
    frames = []

    for label, group in data.items():
        # skip scores that does not have any events due to split
        if label not in score_data:
            continue

        cont_df = pd.DataFrame(to_numpy(group["continuous"]), columns=continuous_names)
        cat_df = pd.DataFrame(to_numpy(group["categorical"]), columns=categorical_names)
        df = pd.concat([cont_df, cat_df], axis=1)

        dataset_name, pid = label[0], label[1]
        df["dataset"] = dataset_name
        df["pid"] = pid

        df["event_id"] = to_numpy(group["event_id"])

        # filter by apply fold_index
        score_data_index = score_data[label]["fold_index"]
        df_fold = df.loc[score_data_index]
        frames.append(df_fold)
    return pd.concat(frames, ignore_index=True)

def build_scores_frame(
    data: dict,
    score_names: list[str],

    ) -> pd.DataFrame:
    """
    Flatten the dict into one Pandas DataFrame with named feature columns + a label column.
    The label columns is taken from names and should represent the order in the tensor.

    Args:
    data (dict): Expect Torch Tensors to be of form: {pid: {"continuous": torch.Tensor, "categorical": torch.Tensor}}
    continuous_names (list[str]):  Column names for the continuous tensor, in order.
    categorical_names (list[str]): Column names for the categorical tensor, in order.

    Returns:
        (pd.DataFrame): One row per event, columns = continuous_names + categorical_names + "label".
    """
    frames = []
    score_names = [f"dnn_score_{name}" for name in score_names]
    for pid, group in data.items():
        df = pd.DataFrame(to_numpy(group["scores"]), columns=score_names)
        df["fold_index"] = to_numpy(group["fold_index"])
        df["normalization_weights"] = to_numpy(group["normalization_weights"])
        df["event_weights"] = to_numpy(group["product_of_weights"])
        df["evaluation_mask"] = to_numpy(group["evaluation_mask"])
        df["event_id"] = to_numpy(group["event_id"])
        df["label"] = str(pid)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
