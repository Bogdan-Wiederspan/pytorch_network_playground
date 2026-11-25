from __future__ import annotations

import functools
import os
import pathlib

from bbTT.monitoring.logger.logger import get_logger

logger_inst = get_logger(__name__)


def find_datasets(dataset_patterns: list[str], year_patterns: list[str], *, file_type: str = "root", verbose=True):
    """
    Find all files in variable ${INPUT_DATA_DIR} by using glob patterns following <year_pattern>/<dataset_pattern>/*.<file_type>.
    The result is as a dictionary of form: {dataset_name : [file_paths]}.
    Duplicates in *year_patterns* and *dataset_patterns* will not affect the result.

    Args:
        dataset_patterns (list[str]): Pattern describing our nano AODs, ex: ['hh_ggf_hbb_htt_kl*_kt1*',]
        year_pattern (list[str]): pattern describing the years e.g. "2*pre" -> 22pre, 23pre
        file_type (str, optional): File extension . Defaults to "root".

    Returns:
        dict: dictionary with dataset names as keys and list of file paths as values
    """

    # precautions: get dir, wrap strings, set pattern
    logger_inst.info("Start searching for datasets:")
    if (data_dir := os.environ.get("INPUT_DATA_DIR", None)) is None:
        raise ValueError("Environment variable INPUT_DATA_DIR not set! Source setup.sh")

    data_dir = pathlib.Path(data_dir)

    if not data_dir.exists():
        raise ValueError(
            f"Given INPUT_DATA_DIR {data_dir} does not exist!\n"
            "Check if directory is named correctly in 'config.sh' or does exist at all"
        )

    if isinstance(dataset_patterns, str):
        dataset_patterns = [dataset_patterns]
    if isinstance(year_patterns, str):
        year_patterns = [year_patterns]

    file_pattern = f"*.{file_type}"

    # resolve year pattern and remove duplicates
    years = []
    for year_pattern in year_patterns:
        years += [year.name for year in data_dir.glob(f"{year_pattern}")]
    years = sorted(set(years))

    data = {}
    missing = []
    for year in years:
        data[year] = {}
        for dataset_patter in dataset_patterns:
            # sort: glob order depends on the filesystem, which makes dataset order
            # (and thus event order downstream) machine-dependent
            datasets = sorted(data_dir.glob(f"{year}/{dataset_patter}"))

            if len(datasets) == 0:
                raise ValueError(f"dataset pattern {dataset_patter} for {year} resulted in 0 datasets")

            for dataset in datasets:
                files = sorted(map(str, pathlib.Path(dataset).glob(file_pattern)))
                if len(files) == 0:
                    logger_inst.critical(f"{dataset} has 0 files")
                    missing.append(dataset)
                if verbose:
                    size = round(sum(os.path.getsize(f) for f in files) / (1024**2), 2)
                    logger_inst.debug(f"+{len(files)} files | size {size} MB | {year}/{dataset.name}")
                data[year][dataset.name] = files
    if not data:
        raise ValueError("No datasets found with given patterns")

    if missing:
        missing_msg = "\n\t".join(missing)
        raise ValueError(f"following datasets has 0 files:\n{missing_msg}")

    # merge over years era information is not needed
    merged_over_era_data = {}
    for era in list(data.keys()):
        dataset_dict = data.pop(era)
        for dataset, files in dataset_dict.items():
            if dataset not in merged_over_era_data:
                merged_over_era_data[dataset] = []
            merged_over_era_data[dataset].extend(files)

    return merged_over_era_data


@functools.lru_cache(maxsize=None)
def cached_find_flat_datasets(dataset_pattern: tuple[str, ...], file_type: str = "parquet") -> dict[str, list[str]]:
    """
    Find all files of a layout without era level: ``${INPUT_DATA_DIR}/<dataset_pattern>/*.<file_type>``.
    This is the layout of the parquet files of Marcel's comparison. Memoized like ``cached_find_datasets``.

    Args:
        dataset_pattern (tuple[str, ...]): Glob patterns of the dataset directories, e.g. ("tt_sl", "dy_*_1j").
        file_type (str, optional): File extension. Defaults to "parquet".

    Returns:
        dict[str, list[str]]: Dataset name -> sorted file paths.

    Raises:
        ValueError: If INPUT_DATA_DIR is not set or does not exist, a pattern matches no directory
            or a matched directory holds no file.
    """
    if (data_dir := os.environ.get("INPUT_DATA_DIR", None)) is None:
        raise ValueError("Environment variable INPUT_DATA_DIR not set! Source setup.sh")
    data_dir = pathlib.Path(data_dir)
    if not data_dir.exists():
        raise ValueError(f"Given INPUT_DATA_DIR {data_dir} does not exist!")

    data: dict[str, list[str]] = {}
    for pattern in dataset_pattern:
        # sort: glob order depends on the filesystem, the event order downstream must not
        datasets = sorted(data_dir.glob(pattern))
        if not datasets:
            raise ValueError(f"dataset pattern {pattern} resulted in 0 datasets in {data_dir}")
        for dataset in datasets:
            files = sorted(map(str, dataset.glob(f"*.{file_type}")))
            if not files:
                raise ValueError(f"dataset {dataset} has 0 {file_type} files")
            data[dataset.name] = files
    return data


@functools.lru_cache(maxsize=None)
def cached_find_datasets(
    dataset_pattern: tuple[str, ...],
    year_pattern: tuple[str, ...],
    file_type: str = "root",
) -> tuple[str, ...]:
    """
    Memoized wrapper around the expensive `find_datasets` call.

    `DataConfig` is often instantiated more than once (e.g. once per module that imports it).
    Each time re-triggering a full dataset lookup, which take time.
    Cache prevents this.

    Returns:
        tuple[str, ...]: dataset file paths (tuple, not list, so the result stays hashable/cacheable).
    """
    return find_datasets(dataset_pattern, year_patterns=year_pattern, file_type=file_type, verbose=False)

def structure_datasets_after_eras(config) -> dict[dict[str, list[str]]]:
    """
    Helper to divide dataset paths from glob to an era like structure:
        Start: {dataset: [paths to all mixed eras]}
        End: {era: {dataset: [paths]}}

    Args:
        config (DatasetConfig): Excepted to be a DatasetConfig Object

    Returns:
        dict[dict[str, list[str]]]: Dictionary with where dataset paths are sorted by era.
    """
    datasets = config.datasets

    # the parquet layout has no era level, all files belong to the single configured era
    if config.file_type == "parquet":
        return {config.eras[0]: {dataset: list(paths) for dataset, paths in datasets.items()}}

    era_datasets = {era: {} for era in config.eras}
    for dataset, paths in datasets.items():
        for path in paths:
            # path is ${INPUT_DATA_DIR}/ERA/dataset/path
            parts = path.split("/")
            era = parts[-3]

            if dataset not in era_datasets[era]:
                era_datasets[era][dataset] = []

            era_datasets[era][dataset].append(path)
    return era_datasets
