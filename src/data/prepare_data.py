import awkward as ak
import numpy as np
import torch

from collections import defaultdict
import torch.utils.data as t_data
from src.data.load_data import load_data
from src.data.datasets import EraDataset, EraDatasetSampler
from src.utils.logger import get_logger

logger = get_logger(__name__)


def awkward_to_torch(array, columns, dtype):
    array = torch.from_numpy(np.stack([array[col].to_numpy() for col in columns], axis=1))
    if dtype is not None:
        array = array.to(dtype)
    return array


def filter_datasets(
    events: dict[dict[ak.Array]],
    feature: str="process_id"
    ) -> dict[dict[ak.Array]]:
    """
    Filter *events* by apply mask to *feature* and returns an dictionary of structure {year: {feature: array}}

    Args:
        events (dict[dict[ak.Array]]): events of structure {year:{dataset : array}}

    Returns:
        return dict[dict[ak.Array]]: events of structure {year: {feature: array}}
    """
    # events of structure {year:{dataset : array}}
    arrays_per_year_per_feature = defaultdict(dict)
    for year, datasets in events.items():
        array_per_feature_collection = defaultdict(list)
        for dataset, array in datasets.items():
            # get unique features, filter array by feature and store in collection
            # {feature : array[feature_mask]}
            for unique_feature in np.unique(array[feature].to_numpy()):
                feature_mask = array[feature] == unique_feature
                filtered_array = array[feature_mask]
                array_per_feature_collection[int(unique_feature)].append(filtered_array)

        for _feature, list_of_arrays in array_per_feature_collection.items():
            array = ak.concatenate(list_of_arrays)
            logger.info(f"{year} | {_feature} | {len(array)} events")
            arrays_per_year_per_feature[year][_feature] = array
    return arrays_per_year_per_feature


def get_sum_of_weights(array):
    # era : {process_id : arrray}
    weights = {}
    for year, pid in array.items():
        weights[year] = {}
        for process_id, arr in pid.items():
            weights[year][process_id] = ak.sum(arr.normalization_weight)
    return weights

def prepare_data(datasets, eras, input_columns, target_columns, dtype=torch.float32, file_type="root"):
    events = load_data(
        datasets,
        eras,
        file_type=file_type,
        columns=input_columns,
        )

    events = filter_datasets(events)
    from IPython import embed; embed(header="string - 265 in load_data.py ")
    # convert to torch tensors and create EraDatasets objects
    EraDatasetManager = EraDatasetSampler(None, batch_size=32)
    for era, pid in events.items():
        c = 0
        for process_id, arr in pid.items():
            if c == 2:
                break
            # from IPython import embed; embed(header="RR - 273 in load_data.py ")
            inputs = awkward_to_torch(arr, input_columns, dtype)
            target = awkward_to_torch(arr, target_columns, dtype)
            weight = torch.tensor(ak.sum(arr.normalization_weight), dtype = dtype)
            era_dataset = EraDataset(
                inputs=inputs,
                target=target ,
                weight=weight ,
                name=process_id,
                era=era,
            )
            EraDatasetManager.add_dataset(era_dataset)
            logger.info(f"Add dataset {process_id} of era {era}")
            c +=1
    return EraDatasetManager

if __name__ == "__main__":
    # test load_dataset
    datasets = ["dy_m50toinf_*j_amcatnlo","tt_dl*"]
    eras = ["22pre", "23pre"]
    input_columns = ["event", "process_id"]
    target_columns = ["target"]
    # filter process after process_id
    e = prepare_data(datasets, eras, input_columns, target_columns, dtype=torch.float32, file_type="root")
    from IPython import embed; embed(header="BEFORE - 288 in load_data.py ")

    input_data = awkward_to_torch(events, ["event", "process_id"], dtype=torch.float32)
    target_data = awkward_to_torch(events, ["target"], dtype=torch.float32)
    # sum_of_weights = ak.full_like(array.normalization_weight, ak.sum(array.normalization_weight))
    # weights = awkward_to_torch(events, ["sum_of_weights"], dtype=torch.float32)
    from IPython import embed; embed(header="Debugger to test the functions - in load_data.py ")

    sampler = t_data.WeightedRandomSampler(weights.flatten(), len(weights))
