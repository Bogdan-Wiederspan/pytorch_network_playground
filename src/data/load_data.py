
import numpy as np
import awkward as ak
import torch
import uproot
import os
import pathlib
from typing import Union

from data.utils import depthCount
from utils.logger import get_logger
from data.cache import DataCacher

logger = get_logger(__name__)

def find_datasets(dataset_patterns: list[str], year_patterns: list[str], *, file_type: str="root", verbose=True):
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
    if (data_dir := os.environ.get("INPUT_DATA_DIR", None)) is None:
        raise ValueError("Environment variable INPUT_DATA_DIR not set! Source setup.sh")

    if isinstance(dataset_patterns, str):
        dataset_patterns = [dataset_patterns]
    if isinstance(year_patterns, str):
        year_patterns = [year_patterns]

    file_pattern = f"*.{file_type}"

    # resolve year pattern and remove duplicates
    years = []
    for year_pattern in year_patterns:
        years += [year.name for year in pathlib.Path(data_dir).glob(f"{year_pattern}")]
    years = sorted(set(years))

    data = {}
    missing = []
    for year in years:
        data[year] = {}
        for dataset_patter in dataset_patterns:
            datasets = list(pathlib.Path(data_dir).glob(f"{year}/{dataset_patter}"))

            if len(datasets) == 0:
                raise ValueError(f"dataset pattern {dataset_patter} for {year} resulted in 0 datasets")

            for dataset in datasets:
                files = sorted(map(str, pathlib.Path(dataset).glob(file_pattern)))
                if len(files) == 0:
                    logger.warning(f"{dataset} has 0 files")
                    missing.append(dataset)
                if verbose:
                    size = round(sum(os.path.getsize(f) for f in files) / (1024**2),2)
                    logger.info(f"+{len(files)} files | size {size} MB | {year}/{dataset.name}")
                data[year][dataset.name] = files
    if not data:
        raise ValueError("No datasets found with given patterns")

    if missing:
        raise ValueError(f"following datasets has 0 files:\n{"\n\t".join(missing)}")

    # merge over years era information is not needed
    merged_over_era_data = {}
    for era in list(data.keys()):
        dataset_dict = data.pop(era)
        for dataset, files in dataset_dict.items():
            if dataset not in merged_over_era_data:
                merged_over_era_data[dataset] = []
            merged_over_era_data[dataset].extend(files)
    return merged_over_era_data


def root_to_numpy(
    files_path: Union[list[str],str],
    branches: Union[list[str], str, None]=None,
    cut: list[str]=None,
    ) -> ak.Array:
    """
    Load all root files in *files_path* and return them as a single awkward array.
    If only certain branches are needed, they can be specified in *branches*.
    To prevent loading unnecessary data a list of *cut*s can be added.

    Args:
        files_path (list[str], str): list of root files or single root file
        branches (list[str], str, optional): branches that should be loaded e.g. ["events", "run"]. If None loads all branches . Defaults to None.

    Returns:
        ak.Array: awkward array containing all data from the root files

    """
    if depthCount(branches) > 1 and branches is not None:
        raise ValueError("branches must be a flat list")

    # set of branches that are always extracted from root files
    # purpose -> fields:
    #   process_id -> filtering of subphasespaces
    #   normalization_weight -> used as sample weight for batch
    #   used for baseline cut -> tau2_isolated, lepton_os, channel_id
    #   event number used for k-fold splitting -> event
    #   oversampling weight that defines the fraction within batch -> normalization_weight
    meta_fields = {"process_id", "tau2_isolated", "leptons_os", "channel_id", "event", "normalization_weight"}

    # training and evaluation phase space are not the same
    # a transfer weight can be calculated using the product of these weights
    weights = {
    "normalized_pdf_weight","normalized_murmuf_weight","normalized_pu_weight",
    "normalized_isr_weight","normalized_fsr_weight","normalized_njet_btag_weight_pnet",
    "electron_id_weight","electron_reco_weight","muon_id_weight","muon_iso_weight",
    "tau_weight","trigger_weight", "dy_weight","top_pt_weight"
    }
    all_branches = set(branches).union(meta_fields)

    # handling cuts
    # by default all analysis has a base cut applied
    # if further cuts are desired, cut is applied on top of baseline cut
    baseline_cuts = [
        "(tau2_isolated == 1)",
        "(leptons_os == 1)",
        "((channel_id == 1) | (channel_id == 2) | (channel_id == 3))",
    ]

    if isinstance(cut, str):
        cut = [cut]
    if cut is None:
        cut = []
    final_cut = "&".join(baseline_cuts + cut)

    # load root files and combine the array to continogus arrays
    if isinstance(files_path, str):
        files_path = [files_path]

    arrays = []
    for file_path in files_path:
        logger.info(f"loading: {file_path}")
        with uproot.open(file_path, object_cache=None, array_cache=None) as file:
            # loading of data is split into 3 steps: all_common, weights, evaluation phasespace
            # reason for 2 is that weights are not present in all datasets
            # e.x. muon_id_weight does not exist in dy datasets, thus require filtering
            # reason for 3: masks that are necessary later are calculated here
            tree = file["events"]

            # step 1
            all_branches_array = tree.arrays(all_branches, library="ak", cut=final_cut)

            # step 2
            weights_in_root_file = set(tree.keys()).intersection(weights)
            weights_arrays = tree.arrays(weights_in_root_file, library="ak", cut=final_cut)

            combined_weight = 1
            for weight in weights_in_root_file:
                combined_weight = combined_weight * weights_arrays[weight]
            all_branches_array["combined_weight"] = combined_weight

            # step 3
            di_tau_mask, di_bjet_mask, bjet_mask = res1b_and_res2b_phase_space_mask(
                uproot_file=tree,
                year=pathlib.Path(file_path).parents[1].stem,
                cut=final_cut,
                suffix=("res_dnn_pnet" if branches[0].startswith("res_dnn_pnet") else "reg_dnn_moe"))
            all_branches_array["bjet_mask"] = bjet_mask
            all_branches_array["di_tau_mask"] = di_tau_mask
            all_branches_array["di_bjet_mask"] = di_bjet_mask

            arrays.append(all_branches_array.to_numpy())
    arrays = np.concatenate(arrays, axis=0)
    return arrays


def res1b_and_res2b_phase_space_mask(uproot_file: str, year: list[str], cut: list[str], suffix: str="res_dnn_pnet"):
    """
    Calculates Masks to get into our evaluation phase space.
    Open *uproot_file* by specific *year*, apply  base *cut* and depending on the producer add a *suffix* to fields in root file.
    Definition of mask is defined in https://github.com/uhh-cms/hh2bbtautau/blob/master/hbt/categorization/default.py#L206-L240

    Args:
        uproot_file (str): Uproot opened root file
        year (list[str]): Year string e.g. "22pre"
        cut (list[str]): Base cut to be applied, should match the one used in root_to_numpy
        suffix (str, optional): Suffix for fields in uproot file. Defaults to "res_dnn_pnet".

    Returns:
        ak.array: Masks for di_tau_mass_window, di_bjet_mass_window, bjet
    """
    # as taken from https://github.com/uhh-cms/hh2bbtautau/blob/master/hbt/config/configs_hbt.py#L1252
    def particle_net_wp(year, wp_level="medium"):
        particle_net_wp = {
            "loose": {"22pre": 0.047, "22post": 0.0499, "23pre": 0.0358, "23post": 0.0359, "2024": None}[year],
            "medium": {"22pre": 0.245, "22post": 0.2605, "23pre": 0.1917, "23post": 0.1919, "2024": None}[year],
            "tight": {"22pre": 0.6734, "22post": 0.6915, "23pre": 0.6172, "23post": 0.6133, "2024": None}[year],
            "xtight": {"22pre": 0.7862, "22post": 0.8033, "23pre": 0.7515, "23post": 0.7544, "2024": None}[year],
            "xxtight": {"22pre": 0.961, "22post": 0.9664, "23pre": 0.9659, "23post": 0.9688, "2024": None}[year],
        }
        return particle_net_wp[wp_level]
    # load the necessary events with applied cut from baseselection
    # all particles are pre rotate relative to visible tau system
    b_tag_wp = particle_net_wp(year, "medium")
    leptons_fields = [f"{suffix}_vis_tau{num}_{kin}" for num in ("1", "2") for kin in ("px", "py", "pz", "e")]
    hhbjet_fields = ["HHBJet_mass", "HHBJet_btagPNetB"]
    di_bjet_fields = [f"{suffix}_bjet{num}_{kin}" for num in ("1", "2") for kin in ("px", "py", "pz", "e")]

    events = uproot_file.arrays(leptons_fields + hhbjet_fields + di_bjet_fields, library="ak", cut=cut)
    ### masks
    # tau mass window
    l_px = events[f"{suffix}_vis_tau1_px"] + events[f"{suffix}_vis_tau2_px"]
    l_py = events[f"{suffix}_vis_tau1_py"] + events[f"{suffix}_vis_tau2_py"]
    l_pz = events[f"{suffix}_vis_tau1_pz"] + events[f"{suffix}_vis_tau2_pz"]
    l_e = events[f"{suffix}_vis_tau1_e"] + events[f"{suffix}_vis_tau2_e"]

    # since no coffee behavior, calculate mass by manually from 4 vector
    di_tau_mass = (l_e**2 - (l_px**2 + l_py**2 + l_pz**2))**0.5
    di_tau_mass_window_mask = (
        (di_tau_mass >= 15) &
        (di_tau_mass <= 130)
    )

    # have atleast 1 bjet
    bjet_mask = ak.sum(events.HHBJet_btagPNetB > b_tag_wp, axis=1) >= 1

    # wrong
    # di_bjet_mass = ak.sum(events.HHBJet_mass, axis=1)

    b_px = events[f"{suffix}_bjet1_px"] + events[f"{suffix}_bjet2_px"]
    b_py = events[f"{suffix}_bjet1_py"] + events[f"{suffix}_bjet2_py"]
    b_pz = events[f"{suffix}_bjet1_pz"] + events[f"{suffix}_bjet2_pz"]
    b_e = events[f"{suffix}_bjet1_e"] + events[f"{suffix}_bjet2_e"]
    di_bjet_mass = (b_e**2 - (b_px**2 + b_py**2 + b_pz**2))**0.5

    di_bjet_mass_window_mask = (
        (di_bjet_mass >= 40) &
        (di_bjet_mass <= 270)
    )

    return di_tau_mass_window_mask, di_bjet_mass_window_mask, bjet_mask


def load_data(datasets, columns: Union[list[str],str, None]=None, cuts: Union[list[str], None]=None):
    """
    Loads data with given *file_type* in given a *dataset_pattern* and *year_pattern*. If only certain columns are needed, they can be specified in *columns*.
    The data sorted by year and dataset name is returned as a nested dictionary in awkward format.

    Args:
        dataset_patter (str): pattern describing dataset e.g. "dy_*" -> all drell-yan datasets
        columns (list[str], str, optional): columns that should be lodead e.g. ["events", "run"]. If None loads all columns . Defaults to None.
        cuts (list[str]): list of cuts to be applied on top of baseline cut, which are defined in root_to_numpy. Defaults to None.

    Returns:
        dict: {year:{pid: List(Ids)}}
    """
    def load_data_per_process_id(dataset_paths, branches, cut=None):
        # helper to load root data into a dictionary of form:
        # {year:{dataset : array}}, where array is a structured array
        def sort_by_process_id(array):
            # helper to extract data by process id and group them by this
            pids = array["process_id"]
            unique_ids = np.unique(pids)
            p_array = {}
            for uid in unique_ids:
                mask = pids == uid
                p_array[int(uid)] = array[mask]
            return tuple(p_array.items())
        data = {}
        for dataset, files in dataset_paths.items():
            events = root_to_numpy(files, branches=branches, cut=cut)
            p_arrays = sort_by_process_id(events)
            for pid, p_array in p_arrays:
                uid = (dataset[:2],pid)
                if uid not in data:
                    data[uid] = []
                data[uid].append(p_array)
                logger.debug(f"{dataset} | PID: {pid} | {len(p_array)}")
        return data

    def merge_per_pid(data):
        # helper to merge all process_ids together to continous array
        keys = list(data.keys())

        logger.info("Start merging arrays per process id")
        for i, uid in enumerate(keys):
            logger.debug(f"{i}/{len(keys)}: {uid}")
            arrays = data.pop(uid)
            concat = np.concatenate(arrays, axis=0)
            data[uid] = concat
        return data
    data = load_data_per_process_id(datasets, branches=list(columns), cut=cuts)
    data = merge_per_pid(data)
    return data


def handle_weights_and_convert_to_torch(events: np.array, continous_features: list[str], categorical_features: list[str], dtype: torch.dtype=None):
    """
    Calculates final weights, extract masks aswell as extract all *continous_features* and *categorical_features* from structured numpy array *events*.
    Converts all arrays to torch tensors and returns a dictionary containing these.

    Args:
        events (np.array): _description_
        continous_features (list[str]): List of continous features to be extracted
        categorical_features (list[str]): List of categorical features to be extracted
        dtype (torch.dtype, optional): Torch dtype. Defaults to None.
    """

    def filter_nan_mask(array, features, uid):
        masks = []
        for f in features:
            mask = np.isnan(array[f])
            masks.append(mask)
        event_mask = np.logical_or.reduce(masks)
        num_filter = np.sum(event_mask)
        if num_filter:
            logger.info(f"Filter {num_filter} Nans out form {uid}")
        return ~event_mask

    for uid in list(events.keys()):
        arr = events.pop(uid)

        # filter all nans out
        event_mask = filter_nan_mask(arr, continous_features + categorical_features, uid),
        arr = arr[event_mask]

        # if resulting tensor is empty just skip
        if arr.size == 0:
            logger.info(f"Skipping {uid} due to zero elements")
            continue

        # combine columns from struct numpy and convert to torch tensor
        continous_tensor, categorical_tensor = [
            torch.from_numpy(np.stack([arr[feature] for feature in features], axis=1))
            for features in (continous_features,categorical_features)
            ]
        # handling weights and convert to torch tensors
        # single numbers cant be converted by using from_numpy thus have to be wrapped in array
        final_mask = arr["bjet_mask"] & arr["di_tau_mask"] & arr["di_bjet_mask"]
        # total_bjet_weight = torch.tensor(np.sum(arr["combined_weight"][arr["bjet_mask"]]))
        # total_di_tau_weight = torch.tensor(np.sum(arr["combined_weight"][arr["di_tau_mask"]]))
        # total_di_bjet_weight = torch.tensor(np.sum(arr["combined_weight"][arr["di_bjet_mask"]]))
        total_evaluation_weight = torch.tensor(np.sum(arr["combined_weight"][final_mask]))

        # some arrays have negative strides for some reason, which torch cannot handle -> cast to contiguous array first
        normalization_weights = torch.tensor(np.ascontiguousarray(arr["normalization_weight"]), dtype=torch.float32)
        sum_of_normalization_weights = torch.sum(normalization_weights)

        product_of_all_weights = torch.tensor(np.ascontiguousarray(arr["combined_weight"]), dtype=torch.float32)
        sum_of_combined_weights = torch.sum(product_of_all_weights)

        # event id is a uint and is stored as uncontiguousarray for some reason after the casting
        event_id = torch.tensor(np.ascontiguousarray(arr["event"]), dtype=torch.int64)

        events[uid] = {
            "continous": continous_tensor,
            "categorical": categorical_tensor,
            "event_id" : event_id,
            "normalization_weights" : normalization_weights,
            "product_of_weights" : product_of_all_weights,

            "total_product_of_weights" : sum_of_combined_weights,
            "total_normalization_weights" : sum_of_normalization_weights,

            "total_evaluation_weight" : total_evaluation_weight,
            # "total_bjet_weight" : total_bjet_weight,
            # "total_di_tau_weight" : total_di_tau_weight,
            # "total_di_bjet_weight" : total_di_bjet_weight,
            "evaluation_mask": torch.tensor(final_mask),
            "mask" : {
                "bjet": arr["bjet_mask"],
                "di_tau": arr["di_tau_mask"],
                "di_bjet": arr["di_bjet_mask"],
                },
        }
    return events


def get_data(config: dict, _save_cache = False, ignore_cache=False) -> dict[torch.Tensor]:
    """
    Main function to combine all steps from loading root files to filter by process ids and finally convert to torch

    Args:
        config (dict): Config as defined in train_config.py
        _save_cache (bool, optional): Save the result of this function as cache, using config as hash. Defaults to False.
        ignore_cache (bool, optional): Rerun loading of data if true, ignoring existing cache. Defaults to False.

    Returns:
        dict[torch.Tensor]: Dictionary with torch tensors
    """
    cacher = DataCacher(config=config)

    # when cache exist load -> it and return the data
    if not ignore_cache and cacher.path.exists():
        events = cacher.load_cache()
    else:
        logger.info("Start loading and filtering of data:")
        events = load_data(
            config["datasets"],
            columns=config["continous_features"] + config["categorical_features"],
            cuts=config["cuts"],
        )
        events = handle_weights_and_convert_to_torch(
            events=events,
            continous_features=config["continous_features"],
            categorical_features=config["categorical_features"],
        )
        logger.info("Done loading data")
        if _save_cache:
            try:
                cacher.save_cache(events)
            except:
                from IPython import embed; embed(header="Saving Cache did not work out - going debugging to manually save \'events\' with \'cacher.save_cache\'")
    return events
