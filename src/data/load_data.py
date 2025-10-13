
import numpy as np
import awkward as ak
import uproot
import os
import pathlib
from typing import Union
from collections import defaultdict
import torch.utils.data as t_data

from src.data.utils import depthCount
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
    era_map = {"22pre": 0, "22post": 1, "23pre": 2, "23post": 3}
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

            # add era encoding:
            era_array = np.full(len(events), era_map[year], np.int32)
            events = ak.with_field(events, era_array, "era")

            data[year][dataset] = events
            logger.info(f"{len(events)} events for {year}/{dataset}")
    return data

if __name__ == "__main__":
    # test load_data
    datasets = ["dy_m50toinf_*j_amcatnlo","tt_dl*"]
    eras = ["22pre", "23pre"]
    input_columns = ["event", "process_id"]
    target_columns = ["target"]
    events = load_data(datasets, eras, file_type="root", columns=input_columns)
    print(events)
    print()
    print(events["22pre"]["dy_m50toinf_0j_amcatnlo"].fields)
    from IPython import embed; embed(header="Test Load Data")
