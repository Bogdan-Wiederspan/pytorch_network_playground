from collections import defaultdict

import numpy as np
import torch

from bbTT.monitoring.logger.logger import get_logger

logger_inst = get_logger(__name__)

class WeightAggregator():

    def __init__(self, events, indices):
        """
        Accumulates and handle process weights from events.

        Args:
            events (_type_): _description_
            indices (_type_): _description_
        """
        self.weights = self.calculate_process_weights(events, indices)
        self.summary_statistic_of_processes = self.calculate_summary_statistics_from_process_weights(self.weights)

    def identify_pid(self, pid):
        if (pid < 2000):
            return "tt"
        elif (pid > 2000) and (pid < 50000):
            return "hh"
        else:
            return "dy"

    def process_weights_sum_from_nested_weight(self, weights, first_key, second_key):
        process_weights_sum = {}
        for uid, _weights in weights.items():
            _, pid = uid
            process_name = self.identify_pid(pid)
            if process_name not in process_weights_sum:
                process_weights_sum[process_name] = 0

            process_weights_sum[process_name] += _weights[first_key][second_key]
        return process_weights_sum

    def calculate_process_weights(self, events, indices):
        weights = {}
        for uid, _events in events.items():

            normalization_weights = _events["normalization_weights"]
            product_of_weights = _events["product_of_weights"]

            weights[uid] = {
                "normalization_weights" : {
                    # "per_event": normalization_weights,
                    "whole_sum": torch.sum(normalization_weights),
                    **{f"{i}_sum": torch.sum(normalization_weights[indices[uid][i]]) for i in ("training", "validation", "test")},
                    "evaluation_sum": torch.sum(normalization_weights[_events["evaluation_mask"]]),

                },
                "product_of_weights": {
                    # "per_event": product_of_weights,
                    "whole_sum": torch.sum(product_of_weights),
                    **{f"{i}_sum": torch.sum(product_of_weights[indices[uid][i]]) for i in ("training", "validation", "test")},
                    "evaluation_sum": torch.sum(product_of_weights[_events["evaluation_mask"]]),
                },
            }
        return weights

    def sum_weights_over(self, first_key, second_key):
        weights = self.process_weights_sum_from_nested_weight(self.weights, first_key=first_key, second_key=second_key)
        return weights

    # TODO REMOVE
    def calculate_summary_statistics_from_process_weights(self, weights_per_process=None):
        if weights_per_process is None:
            self.weights_per_process = self.weights
        weights = {
            weight_name: self.process_weights_sum_from_nested_weight(weights_per_process, first_key=weight_name, second_key="whole_sum")
            for weight_name in ("product_of_weights", "normalization_weights")
        }
        return weights

    def __call__(self, kind_weight, phase_space):
        return self.sum_weights_over(kind_weight, phase_space)


def apply_tokenization(expected_inputs, events, categorical_features):
    for uid in list(events.keys()):
        data = events[uid]
        cateogrical_array = data["categorical"]
        map_categorical_features(
            expected_inputs=expected_inputs,
            feature_array=cateogrical_array,
            categorical_features=categorical_features
        )
        events["categorical"] = cateogrical_array
    return events

def map_categorical_features(expected_inputs, feature_array, categorical_features):
    feat_window_start = 0
    feat_window_end = 0
    new_indices = {}
    for cat in categorical_features:
        feat_window_end += len(expected_inputs[cat])
        indices = np.arange(start=feat_window_start, stop=feat_window_end)
        feat_window_start = feat_window_end
        new_indices[cat] = indices

    # get all masks
    for idx, cat in enumerate(categorical_features):
        old_values = expected_inputs[cat]
        new_values = new_indices[cat]
        masks = []
        # get masks
        for value in old_values:
            m = feature_array[:, idx] == value
            masks.append(m)

        # apply masks inplace
        for mask, new_value in zip(masks, new_values):
            feature_array[:, idx][mask] = new_value
    return feature_array

def get_batch_statistics_from_sampler(sampler=None, padding_values=None, features=None, return_dummy=False):
    """
    Calculates the weighted mean and standard deviation over all subphase spaces of a process in *sampler*.
    The data is expected to be of form : {"unique_identifier_tuple": {continuous: arr}, {weight}: arr}.
    The return value is a dictionary of form {"process": (mean, std)}, where mean and std is a tensor of
    form length [features].
    The statistics are evaluated without *padding_values*.
    If values are used only for init of Standardization Layer turn on *return_dummy*, which return a mean and std tensor, corresponding to 0 and 1.

    Args:
        sampler (dict): Dictionary over datasets
        padding_values (int, optional): List of padding_values per feature, or single value. Padding value are ignored in the calculation of the statitics. Defaults to None, which means no padding.
        return_dummy (bool): Return dummy values that describe Identity transformation. This does not compute the statistics, and are a good option when one load pretrained weights anyway.
    """
    # when only dummy values for init are necessary return mean = 0 and std as 1
    if return_dummy:
        num_f = len(features)
        mean = torch.zeros(num_f)
        std = torch.ones(num_f)
        logger_inst.info(
            "\nNo normalization statistics are calculated, since return_dummy is True."
            "Returning dummy values: mean = 0 and std = 1"
            )
        return mean, std

    logger_inst.info("Calculate mean and std over all subphase spaces")
    # filter keys after processes
    weighted_means = []
    weighted_vars = []
    features_dict = sampler.continuous()
    weights_dict = sampler.relative_weight()
    sum_of_weights = sum(list(weights_dict.values()))
    # for each process id calculate weighted mean, std for each feature representing batch statistics
    for current_pid_idx, pid in enumerate(features_dict.keys(), start = 1):
        # get array of specific process id - [num_events x num_features]
        array = features_dict[pid]

        # no logging mechanism for flushing on the same line
        logger_inst.info_progress(f"\rcalculating stats for pids: {current_pid_idx}/{len(features_dict)}")

        # mean and var calculation should exclude padding values
        # if padding value shape is 1 -> expand to num_feature or equals num_features
        num_f = array.shape[-1]

        ignore_tensor = torch.tensor(padding_values)
        if ignore_tensor.shape == torch.Size([]):
            ignore_tensor = torch.full((1, num_f), fill_value=padding_values)
        elif ignore_tensor.shape != torch.Size([num_f]):
            raise ValueError(f"Padding values need to be of shape [num_features] or single value, got {ignore_tensor.shape}")

        # create and apply mask to include values
        include_mask = ~(array == ignore_tensor)
        masked_mean = torch.masked.mean(input=array, mask=include_mask, dim=0, dtype=torch.float64)
        masked_var = torch.masked.var(input=array, mask=include_mask, dim=0, dtype=torch.float64)
        if torch.any(masked_mean.isnan()):
            from IPython import embed
            embed(header=f"{pid} is nan check feature_array and sampler")

        # weight mean and add to collection
        # pid_weight = weights_dict[pid]
        # weighted_means.append(torch.tensor(f_means) * pid_weight)
        # weighted_vars.append(torch.tensor(f_vars) * pid_weight)
        pid_weight = weights_dict[pid]
        weighted_means.append(masked_mean * pid_weight)
        weighted_vars.append(masked_var * pid_weight)

    # calculate weighted average over uid means and var
    print()
    w_avg_mean  = torch.sum(torch.stack(weighted_means, axis=0), axis = 0) / sum_of_weights
    w_avg_var = torch.sum(torch.stack(weighted_vars, axis=0), axis = 0) / sum_of_weights
    if features:
        msg = []
        for f_name, f_mean, f_var in zip(features, w_avg_mean, w_avg_var):
            msg.append(f"{f_name:<30}: mean:{f_mean:>10.4} var:{f_var:>10.4}")
        logger_inst.debug("\n" + "\n".join(msg))
    return w_avg_mean, w_avg_var.sqrt()


def get_batch_statistics(events=None, padding_value=0):
    """
    Calculates the weighted mean and standard deviation over all subphase spaces of a process in *events*.
    The data is expected to be of form : {"unique_identifier_tuple": {continuous: arr}, {weight}: arr}.
    The return value is a dictionary of form {"process": (mean, std)}, where mean and std is a tensor of
    form length [features].
    The statistics are evaluated without *padding_value*.

    Args:
        events (dict): Dictionary over datasets
        padding_value (int, optional): Ignored value in the calculation of the statitics. Defaults to 0.
    """
    logger_inst.info("Start calcuation of mean and std over all subphase spaces.")
    # filter keys after processes
    means = []
    stds = []
    weights = []
    for uid, arrays in events.items():
        # reshape to feature x events

        arr_features = arrays["continuous"].transpose(0,1)
        weights.append(arrays["weight"])
        # go throught each feature axis and calculate statitic per feature
        f_means, f_stds = [], []
        for f in arr_features:
            padding_mask = (f == padding_value)
            masked_array = f[~padding_mask]
            masked_mean = masked_array.mean(axis=0)
            masked_std = masked_array.std(axis=0)
            if torch.isnan(masked_mean):
                from IPython import embed
                embed(header=f"{uid} is nan check f and events")

            f_means.append(masked_mean)
            f_stds.append(masked_std)
        means.append(f_means)
        stds.append(f_stds)
    means = torch.tensor(means)
    stds = torch.tensor(stds)
    weights = torch.tensor(weights).reshape(-1,1)

    # resulting in a weight of form [features]
    denom = torch.sum(weights)
    w_avg_mean  = torch.sum((means * weights), axis=0) / denom
    w_avg_std = torch.sum((stds * weights), axis=0) / denom
    return w_avg_mean, w_avg_std


def get_batch_statistics_per_dataset(events, padding_value=0):
    """
    Calculates the weighted mean and standard deviation over all subphase spaces of a process in *events*.
    The data is expected to be of form : {"unique_identifier_tuple": {continuous: arr}, {weight}: arr}.
    The return value is a dictionary of form {"process": (mean, std)}, where mean and std is a tensor of
    form length [features].
    The statistics are evaluated without *padding_value*.

    Args:
        events (dict): Dictionary over datasets
        padding_value (int, optional): Ignored value in the calculation of the statitics. Defaults to 0.
    """
    logger_inst.info("Start calculation of mean and std over all subphase spaces")
    # filter keys after processes
    keys_per_process = defaultdict(list)
    for uid in events.keys():
        (ds_type, pid) = uid
        keys_per_process[ds_type].append(uid)

    stats = {}
    for process_type, uids in keys_per_process.items():
        means = []
        stds = []
        weights = []
        for uid in uids:
            f_means, f_stds = [], []
            # reshape to feature x events
            arr_features = events[uid]["continuous"].transpose(0,1)
            weights.append(events[uid]["weight"])

            # go throught each feature axis and calculate statitic per feature

            for f in arr_features:
                padding_mask = f == padding_value
                f_means.append(f[~padding_mask].mean(axis=0))
                f_stds.append(f[~padding_mask].std(axis=0))

                if torch.isnan(f[~padding_mask].mean(axis=0)):
                    from IPython import embed
                    embed(header="See which feature is nan")
            means.append(f_means)
            stds.append(f_stds)
        means = torch.tensor(means)
        stds = torch.tensor(stds)
        weights = torch.tensor(weights).reshape(-1,1)

        # resulting in a weight of form [features]
        nom = torch.sum((means * weights), axis=0)
        denom = torch.sum(weights)
        stats[process_type] = nom / denom
    return stats

def k_fold_indices(event_id, c_fold, k_fold, seed, test=False):
    """
    Creates idicies for training and validation from *k_fold*, where *c_fold* is the test fold.
    The indicies are permutated using *seed*. If *test* is True, the test set is returned if False the k-1 folds are returned.

    Args:
        num_events (int): Number of events for which k-folds should be created
        c_fold (int): Current fold, this fold is returned when *test* is True
        k_fold (int): Number of folds
        seed (int): Seeds used for random permutation
    """
    # true => test folds, false => train and validation folds
    # if no kfold is wished than set test to 0 and return everyrthing
    if k_fold == 0:
        raise ValueError(f"Can't do k-fold with desired k_fold of {k_fold}, needs to be > 0")
    test_fold_mask = event_id % k_fold == c_fold
    indices = torch.arange(len(event_id))

    if test:
        sub_event_id = indices[test_fold_mask]
    else:
        sub_event_id = indices[~test_fold_mask]
    # apply mask and randomize according to given seed
    randomized = torch.randperm(len(sub_event_id), generator=torch.Generator().manual_seed(seed))
    return sub_event_id[randomized]

def split_array_to_train_and_validation(array, trainings_proportion=0.75):
    """
    Splits given *array* into *trainings_proportion* train and (1 - *trainings_proportion*) validation parts.

    Args:
        array (torch.Tensor, numpy.Array): flat torch or numpy array
        trainings_proportion (float, optional): Relative proportion of the resulting trainings array. Defaults to 0.75.

    Returns:
        tuple (torch.Tensor, numpy.Array): Tuple of trainings and validation array
    """
    if (trainings_proportion > 1) or (trainings_proportion < 0):
        raise ValueError(f"Split fraction is {trainings_proportion} but needs to be in range of 0 and 1")
    train_length = int(round((len(array) * trainings_proportion)))
    t_idx = array[:train_length]
    v_idx = array[train_length:]
    return t_idx, v_idx

def split_k_fold_into_training_and_validation(events_dict, c_fold, k_fold, seed, train_ratio=0.75, return_test=False):
    """
    Takes *events_dict* where continuous and categorical data and split these into *k_fold* where *c_fold* is the holdout test seed.
    A random permutation happens using *seed* and in the end the k-1 folds are split into training
    and validation data using a ration of *train_ratio*.

    Args:
        events_dict (torch.tensor): Dictionary of continuous and categorical Tensors
        c_fold (int): Current fold, this fold is returned when *test* is True
        k_fold (int): Number of folds
        seed (int): Seeds used for random permutation
        train_ratio (float, optional): Percentage of training data. Defaults to 0.75.

    Returns:
        dict(torch.tensor): Train and validation dictionary
    """
    train, valid = {}, {}
    for uid in list(events_dict.keys()):
        # handle constant values that are independent of splitting, like sum of weights
        array = events_dict.pop(uid)

        # create a copy of the dictionary with constant values
        # otherwise train and valid would point to the same memory
        constant_values = {
            "total_normalization_weights" : array["total_normalization_weights"],
            "total_product_of_weights" : array["total_product_of_weights"]
            }
        train[uid], valid[uid] = constant_values.copy(), constant_values.copy()

        tv_indices = k_fold_indices(array["event_id"], c_fold, k_fold, seed, test=return_test)
        t_idx, v_idx = split_array_to_train_and_validation(tv_indices, train_ratio)
        # splitt arrays into train and validation
        for key in ("continuous", "categorical", "event_id", "normalization_weights", "product_of_weights", "evaluation_mask"):

            arr = array.pop(key)
            # there are multiple masks
            train[uid][key], valid[uid][key] = arr[t_idx], arr[v_idx]


    # edge case split results in empty tensors (due to very low event count) remove these
    # if empty do not save
    for uid in list(train.keys()):
        for d in ("train", "valid"):
            dictionary = locals()[d]
            if (dictionary[uid]["continuous"].numel() == 0):
                logger_inst.warning(f"removed {uid} from {d} since zero elements left after k-fold split")
                dictionary.pop(uid)
    return train, valid
