
from src.data.utils import depthCount
import torch
import numpy as np
import awkward as ak
import uproot
import os
import pathlib
from typing import Union
from collections import defaultdict
import torch.utils.data as t_data

from src.utils.logger import get_logger
logger = get_logger(__name__)


def find_datasets(dataset_patterns: str, year_patterns: str, file_type: str="root"):
    """
    Find all files in ${INPUT_DATA_DIR} by pattern defined by <year_pattern>/<dataset_pattern>/*.file_type and
    return the result as a dictionary with dataset names as keys and list of file paths as values.

    Args:
        dataset_patter (str): _description_
        year_pattern (str): Pattern describing the years
        file_type (str, optional): File extension . Defaults to "root".

    Returns:
        dict: dictionary with dataset names as keys and list of file paths as values
    """
    if (data_dir := os.environ.get("INPUT_DATA_DIR", None)) is None:
        raise ValueError("Environment variable INPUT_DATA_DIR not set! Source setup.sh")

    # wrapp patterns in lists if only being a single string
    if isinstance(dataset_patterns, str):
        dataset_patterns = [dataset_patterns]
    if isinstance(year_patterns, str):
        year_patterns = [year_patterns]

    file_pattern = f"*.{file_type}"

    years = []
    for year_pattern in year_patterns:
        years += [year.name for year in pathlib.Path(data_dir).glob(f"{year_pattern}")]

    data = {}
    for year in years:
        data[year] = {}
        for dataset_patter in dataset_patterns:
            datasets = list(pathlib.Path(data_dir).glob(f"{year}/{dataset_patter}"))

            if len(datasets) == 0:
                logger.warning(f"skip {year} due to 0 datasets")
                continue

            for dataset in datasets:
                files = list(map(str, pathlib.Path(dataset).glob(file_pattern)))
                if len(files) == 0:
                    logger.warning(f"skip {dataset.name} due to 0 files")
                    break
                size = round(sum(os.path.getsize(f) for f in files) / (1024**2),2)
                logger.info(f"+{len(files)} files | size {size} MB | {year}/{dataset.name}")
                data[year][dataset.name] = files
    return data

def root_to_awkward(files_path: Union[list[str],str], branches: Union[list[str], str, None]=None) -> ak.Array:
    """
    Load all root files in *files_path* and return them as a single awkward array.
    If only certain branches are needed, they can be specified in *branches*.

    Args:
        files_path (list[str], str): list of root files or single root file
        branches (list[str], str, optional): branches that should be loaded e.g. ["events", "run"]. If None loads all branches . Defaults to None.

    Returns:
        ak.Array: awkward array containing all data from the root files

    """
    if depthCount(branches) > 1 and branches is not None:
        raise ValueError("branches must be a flat list")

    if isinstance(files_path, str):
        files_path = [files_path]
    arrays = []
    for file_path in files_path:
        with uproot.open(file_path) as file:
            tree = file["events"]
            arrays.append(tree.arrays(branches, library="ak"))
    return ak.concatenate(arrays, axis=0)

def parquet_to_awkward(files_path: Union[list[str],str], columns: Union[list[str], str, None]=None) -> ak.Array:
    """
    Load all parquet files in *files_path* and return them as a single awkward array.
    If only certain columns are needed, they can be specified in *columns*.

    Args:
        files_path (list[str], str): list of root files or single root file
        columns (list[str], str, optional): columns that should be loaded e.g. ["events", "run"]. If None loads all branches . Defaults to None.

    Returns:
        ak.Array: awkward array containing all data from the root files
    """
    if depthCount(columns) > 1 and columns is not None:
        raise ValueError("columns must be a flat list")
    arrays = []
    for file_path in files_path:
        arrays.append(ak.from_parquet(file_path, columns=columns))
    return ak.concatenate(arrays, axis=0)


def awkward_to_torch(array, columns, dtype):
    array = torch.from_numpy(np.stack([array[col].to_numpy() for col in columns], axis=1))
    if dtype is not None:
        array = array.to(dtype)
    return array

# def awkward_to_torch(array: ak.Array, columns: Union[list[str],str], dtype: torch.dtype=None, combine_datasets=False) -> torch.Tensor:
#     """
#     Helper function to convert an ragged awkward *array* to a regulare torch tensor of a specific *dytpe*.
#     Important: the order of *columns* defines the order of the features in the returned tensor.

#     Args:
#         array (ak.Array): awkwary array to be converted
#         columns (list[str]): columns within array that should be converted in the given order of columns
#         dtype (torch.dtype, optional): dtype of the torch tensor. If None, dtype is tried to preserve. Defaults to None.

#     Returns:
#         torch.Tensor: converted torch tensor of given dtype
#     """
#     # loop over columns to get arrays in correct order
#     # remove nested structure by flattening the dictionary
#     from IPython import embed; embed(header="COMBINE - 124 in load_data.py ")
#     if combine_datasets:
#         # combine the different process_id or datasets
#         data = []
#         for era, dataset_or_process_id in array.items():
#             for arr in dataset_or_process_id.values():
#                 data.append(arr)
#         data = ak.concatenate(data, axis=0)

#     def replace_array(array):
#         replace_data = {}
#         for era, pid in array.items():
#             for process_id, arr in pid.items():
#                 replace_data[process_id] = torch.from_numpy(arr.to_numpy())


    # # take only columns in specific order
    # # stack to numpy and convert to torch tensor
    # # preserve dtype if none
    # data = torch.from_numpy(np.stack([data[col].to_numpy() for col in columns], axis=1))
    # if dtype is not None:
    #     data = data.to(dtype)
    # return data

def get_loader(file_type: str, **kwargs):
    """
    Small helper function to get the correct loader function and its configuration based on the file type.

    Args:
        file_type (str): file extension.

    Raises:
        ValueError: if file type is not supported by loader

    Returns:
        func: loader function, configuration dictionary
    """
    match file_type:
        case "root":
            return root_to_awkward, {"branches": kwargs.get("columns", None)}
        case "parquet":
            return parquet_to_awkward, {"columns": kwargs.get("columns", None)}
        case _:
            raise ValueError(f"Unknown file type: {file_type}")

def load_data(dataset_patter: str, year_pattern: str, file_type: str="root", columns: Union[list[str],str, None]=None):
    """
    Loads data with given *file_type* in given a *dataset_pattern* and *year_pattern*. If only certain columns are needed, they can be specified in *columns*.
    The data sorted by year and dataset name is returned as a nested dictionary in awkward format.

    Args:
        dataset_patter (str): pattern describing dataset e.g. "dy_*" -> all drell-yan datasets
        year_pattern (str): pattern describing the years e.g. "2*pre" -> 22pre, 23pre
        file_type (str, optional): file extension. Defaults to "root".
        columns (list[str], str, optional): columns that should be lodead e.g. ["events", "run"]. If None loads all columns . Defaults to None.

    Returns:
        _type_: _description_
    """
    target_map = {"hh" : 0, "dy": 1, "tt": 2}
    num_targets = len(target_map.keys())
    # add weights for resampling
    columns = set(columns)
    if "normalization_weight" not in columns:
        columns.add("normalization_weight")

    loader, config = get_loader(file_type, columns=list(columns))
    datasets = find_datasets(dataset_patter, year_pattern, file_type)
    data = {}
    for year, year_data in datasets.items():
        data[year] = {}
        # add events with structure {dataset_name : events}
        for dataset, files in year_data.items():
            # load inputs
            events = loader(files, **config)

            # add target by dataset name, first 2 letters define the target
            target_value = target_map[dataset[:2]]
            target_array = np.zeros((len(events), num_targets))
            target_array[:, target_value] = 1
            events = ak.with_field(events, target_array, "target")

            data[year][dataset] = events
            logger.info(f"{len(events)} events for {year}/{dataset}")
    return data

def filter_datasets_after_process(events):
    """
    Gets *events* in the structure of {year: {dataset: array}} and filters array by process_id.
    Returns an dictionary of structure {year: {process_id: array}}

    Args:
        events (_type_): _description_

    Returns:
        _type_: _description_
    """
    # events of structure {year:{dataset : array}}
    arrays_per_year_per_id = {}
    for year, datasets in events.items():
        arrays_per_id_collection = {}
        arrays_per_year_per_id[year] = {}
        for dataset, array in datasets.items():
            # {id : array[id_mask]}
            arrays_per_id = filter_array_after_process(array)
            for process_id, array_per_id in arrays_per_id.items():
                if process_id not in arrays_per_id_collection:
                    arrays_per_id_collection[process_id] = []
                arrays_per_id_collection[process_id].append(array_per_id)

        for process_id, list_of_arrays in arrays_per_id_collection.items():
            array = ak.concatenate(list_of_arrays)
            logger.info(f"{year} | {process_id} | {len(array)} events")
            arrays_per_year_per_id[year][process_id] = array
    return arrays_per_year_per_id


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

    events = filter_datasets_after_process(events)
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
    from datasets import EraDataset, EraDatasetSampler

    datasets = ["dy_m50toinf_*j_amcatnlo","tt_dl*"]
    eras = ["22pre", "23pre"]
    input_columns = ["event", "process_id"]
    target_columns = ["target"]
    # filter process after process_id
    from IPython import embed; embed(header="BEFORE - 288 in load_data.py ")
    e = prepare_data(datasets, eras, input_columns, target_columns, dtype=torch.float32, file_type="root")
    input_data = awkward_to_torch(events, ["event", "process_id"], dtype=torch.float32)
    target_data = awkward_to_torch(events, ["target"], dtype=torch.float32)
    # sum_of_weights = ak.full_like(array.normalization_weight, ak.sum(array.normalization_weight))
    # weights = awkward_to_torch(events, ["sum_of_weights"], dtype=torch.float32)
    from IPython import embed; embed(header="Debugger to test the functions - in load_data.py ")

    sampler = t_data.WeightedRandomSampler(weights.flatten(), len(weights))
