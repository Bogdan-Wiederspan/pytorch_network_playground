
import pathlib
from typing import Union

import awkward as ak
import numpy as np
import torch
import uproot

from bbTT.data_handling.cache import DataCacher
from bbTT.data_handling.evaluation_phase_space import res1b_and_res2b_phase_space_mask
from bbTT.data_handling.utils import depthCount, struct_to_group_tensor
from bbTT.monitoring.logger.logger import get_logger

logger_inst = get_logger(__name__)


def load_root_and_convert_to_numpy(
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
    logger_inst.info("Start loading and conversion of root files:")
    if depthCount(branches) > 1 and branches is not None:
        raise ValueError(f"branches must be a flat list but is {depthCount(branches)}-dimensional")

    # set of branches that are always extracted from root files
    meta_fields = {
        "process_id", # filtering of sub-phase spaces
        "event", # event number used for k-fold splitting
        "normalization_weight" # oversampling weight that defines the fraction within batch
    }

    # columns necessary for evaluation masks
    suffix = "res_dnn_pnet" if branches[0].startswith("res_dnn_pnet") else "reg_dnn_moe"
    mask_fields = (
        [f"{suffix}_vis_tau{num}_{kin}" for num in ("1", "2") for kin in ("px", "py", "pz", "e")] +
        ["HHBJet_mass", "HHBJet_btagPNetB"] +
        [f"{suffix}_bjet{num}_{kin}" for num in ("1", "2") for kin in ("px", "py", "pz", "e")]
    )

    # training and evaluation phase space are not the same
    # a transfer weight can be calculated using the product of these weights
    weights = {
        "normalized_pdf_weight",
        "normalized_murmuf_weight",
        "normalized_pu_weight",
        "normalized_isr_weight",
        "normalized_fsr_weight",
        "normalized_njet_btag_weight_pnet",
        "electron_id_weight",
        "electron_reco_weight",
        "muon_id_weight",
        "muon_iso_weight",
        "tau_weight",
        "trigger_weight",
        "dy_weight",
        "top_pt_weight"
    }
    # combine all branches to have only 1 uproot -> ak -> numpy chain
    all_branches = set(branches).union(meta_fields).union(mask_fields)

    # handle edge cases for cuts and file paths
    if isinstance(cut, str):
        cut = [cut]
    cut = "&".join(cut) if cut else None

    if isinstance(files_path, str):
        files_path = [files_path]

    # conversion
    arrays = []
    max_files = len(files_path)
    for current_file, file_path in enumerate(files_path, start=1):
        logger_inst.debug_progress(f"loading file: {current_file}/{max_files}")

        with uproot.open(file_path, object_cache=None, array_cache=None) as file:

            tree = file["events"]
            # some weights are dataset specific and needs to be extracted on uid base
            # e.x. muon_id_weight does not exist in dy datasets
            weights_in_current_file = set(tree.keys()).intersection(weights)
            read_fields = all_branches.union(weights_in_current_file)

            events = tree.arrays(read_fields, library="ak", cut=cut)

            # calculate_monte_carlo_weight
            # this weight combines all corrections, but also over-sampling of mc generator
            combined_weight = events["normalization_weight"]
            for w in weights_in_current_file:
                combined_weight = combined_weight * events[w]
            events["combined_weight"] = combined_weight

            # calculate evaluation phase space mask
            year=pathlib.Path(file_path).parents[1].stem
            di_tau_mask, di_bjet_mask, bjet_mask = res1b_and_res2b_phase_space_mask(
                events=events,
                year=year,
                suffix=suffix
            )

            events["bjet_mask"] = bjet_mask
            events["di_tau_mask"] = di_tau_mask
            events["di_bjet_mask"] = di_bjet_mask

            # drop inputs, and keep only artifacts and results
            keep = set(branches).union(meta_fields).union(
                {"combined_weight", "bjet_mask", "di_tau_mask", "di_bjet_mask"}
            )

            events_np = events[list(keep)].to_numpy()
            del events

            arrays.append(events_np)
    return arrays

def stream_events_by_uid(
    dataset_paths: list[str],
    columns: Union[list[str], str],
    cut: Union[list[str], None]=None,
    flush_threshold_rows: int =1_000_000
    ) -> dict[tuple[str, int], np.typing.ArrayLike]:
    """
    Load root files in *dataset_paths*, extract *columns* with applied *cut* on it.
    The data is then rearranged by their process_id, removing year and era information.
    To reduce peak memory *flush_threshold_rows* can be adjusted. The higher, the bigger is peak memory impact vs. CPU time.
    Returns dictionary with np.arrays where key is (dataset, uid), ex. ('tt', 1100).

    Args:
        dataset_paths (list[str]): pattern describing dataset e.g. "dy_*" -> all drell-yan datasets
        branches (Union[list[str], str]): columns that should be loaded e.g. ["events", "run"]. If None loads all columns . Defaults to None.
        cut (Union[list[str], None], optional): list of cuts to be applied on top of baseline cut, which are defined in root_to_numpy. Defaults to None.
        flush_threshold_rows (int, optional): Number of rows inside array before flush is started. Trade of between CPU and peak memory. Defaults to 1_000_000.

    Returns:
        dict[tuple[str, int], np.typing.ArrayLike]: Dict of arrays where key is tuple of dataset name and id.
    """
    def group_by_process_id(array):
        # helper to extract data by process id and group them by this
        pids = array["process_id"]
        for uid in np.unique(pids):
            yield int(uid), array[pids == uid]

    buffers = {}       # uid -> list of not-yet-merged fragments
    buffered_rows = {}  # uid -> sum of rows currently buffered (unmerged)
    data = {}           # uid -> merged running array

    def flush(uid):
        # helper to flush buffer and integrate into data
        fragments = buffers.get(uid)
        if not fragments:
            return

        # concatenate data or just unpack
        fragment = fragments[0]
        if len(fragments) > 1:
            fragment = np.concatenate(fragments, axis=0)

        # reset buffer
        buffers[uid] = []
        buffered_rows[uid] = 0

        # when uid fresh just add as base, else append fragment to array
        if uid not in data:
            data[uid] = fragment
        else:
            data[uid] = np.concatenate([data[uid], fragment], axis=0)


    for dataset, files in dataset_paths.items():
        for events in load_root_and_convert_to_numpy(files, branches=columns, cut=cut):
            for pid, p_array in group_by_process_id(events):
                uid = (dataset[:2],pid)
                buffers.setdefault(uid, []).append(p_array)
                buffered_rows[uid] = buffered_rows.get(uid, 0) + len(p_array)
                if buffered_rows[uid] >= flush_threshold_rows:
                    flush(uid)
            del events
    # final flush for any remaining buffered fragments:
    for uid in list(buffers.keys()):
        flush(uid)
        logger_inst.debug(f"UID: {uid} | NUM: {len(data[uid])}")
    return data

def filter_and_convert_to_torch(events: np.array, continuous_features: list[str], categorical_features: list[str]):
    """
    Calculates final weights, extract masks aswell as extract all *continuous_features* and *categorical_features* from structured numpy array *events*.
    Converts all arrays to torch tensors and returns a dictionary containing these.

    Args:
        events (np.array): _description_
        continuous_features (list[str]): List of continuous features to be extracted
        categorical_features (list[str]): List of categorical features to be extracted
        dtype (torch.dtype, optional): Torch dtype. Defaults to None.
    """

    def filter_nan_mask(array, features, uid):
        event_mask = np.zeros(array.size, dtype=np.bool)
        for f in features:
            event_mask |= np.isnan(array[f])
        num_filter = np.sum(event_mask)
        if num_filter:
            logger_inst.warning(f"Filtered {num_filter} Nan events from pid {uid}")
        return ~event_mask

    for uid in list(events.keys()):
        arr = events.pop(uid)

        # filter all nans out, when result is empty array, skip whole uid
        event_mask = filter_nan_mask(arr, continuous_features + categorical_features, uid)
        arr = arr[event_mask]

        if arr.size == 0:
            logger_inst.warning(f"Skipping {uid} due to zero elements - which can happen after filtering nans")
            continue

        # extract features
        continuous_tensor = struct_to_group_tensor(arr, continuous_features, dtype=torch.float32)
        categorical_tensor = struct_to_group_tensor(arr, categorical_features, dtype=torch.float32)

        # extract evaluation masks
        masks_tensor = struct_to_group_tensor(arr, ["bjet_mask", "di_tau_mask", "di_bjet_mask"], torch.bool)
        final_mask = masks_tensor[:, 0] & masks_tensor[:, 1] & masks_tensor[:, 2]

        # extract weights
        weights_tensor = struct_to_group_tensor(arr, ["normalization_weight", "combined_weight"], dtype=torch.float32)
        normalization_weights = weights_tensor[:, 0]
        sum_of_normalization_weights = torch.sum(normalization_weights)

        product_of_all_weights = weights_tensor[:, 1]
        sum_of_combined_weights = torch.sum(product_of_all_weights)
        total_evaluation_weight = torch.sum(product_of_all_weights[final_mask])

        # extract meta fields
        event_id = struct_to_group_tensor(arr, ["event"], dtype=torch.int64).flatten()

        events[uid] = {
            "continuous": continuous_tensor,
            "categorical": categorical_tensor,
            "event_id" : event_id,
            "normalization_weights" : normalization_weights,
            "product_of_weights" : product_of_all_weights,

            "total_product_of_weights" : sum_of_combined_weights,
            "total_normalization_weights" : sum_of_normalization_weights,

            "total_evaluation_weight" : total_evaluation_weight,
            "evaluation_mask": final_mask,
            "mask" : {
                "bjet": masks_tensor[:, 0],
                "di_tau": masks_tensor[:, 1],
                "di_bjet": masks_tensor[:, 2],
                },
        }
        del arr
    return events


def get_data(config , save_cache = False, ignore_cache=False, debug_on_cache_failure=True) -> dict[str, torch.Tensor]:
    """
    Main function to combine all steps from loading root files to filter by process ids and finally convert to torch

    Args:
        config (DataClass): Config as defined in train_config.py
        save_cache (bool, optional): Save the result of this function as cache, using config as hash. Defaults to False.
        ignore_cache (bool, optional): Rerun loading of data if true, ignoring existing cache. Defaults to False.

    Returns:
        dict[torch.Tensor]: Dictionary with torch tensors
    """
    cache = DataCacher(config=config)

    # when cache exist load -> it and return the data
    all_events = {}



    for era in sorted(config.eras, key=config.era_size):
        if not ignore_cache and cache.era_exists(era):
            logger_inst.info(f"Loading cached data for era {era}")
            era_events = cache.load_era(era)
            all_events = merge_era_events(all_events, era_events)
            continue

        # start creation of cache
        logger_inst.info(f"Start loading and filtering of data for era {era}")
        era_events = stream_events_by_uid(
            config.datasets,
            columns=config.uproot_continuous_columns + config.uproot_categorical_columns,
            cut=config.uproot_cuts,
            flush_threshold_rows=config.flush_threshold,
        )
        logger_inst.info("Start handling weights and conversion to torch tensors")
        era_events = filter_and_convert_to_torch(
            events=era_events,
            continuous_features=config.uproot_continuous_columns,
            categorical_features=config.uproot_categorical_columns,
        )

        if save_cache:
            try:
                cache.save_era(era=era, events=era_events)
                logger_inst.info(f"Finished cache for era {era}")
            except Exception as e:
                logger_inst.exception(f"Sacing cache for era {era} failed")
                if debug_on_cache_failure:
                    from IPython import embed
                    embed(header=f"{e}\n Saving Cache did not work out - going debugging to manually save \'events\' with \'cacher.save_cache\'")

        all_events = merge_era_events(all_events, era_events)
        del era_events
    return all_events

def merge_era_events(all_events, era_events):
    # incremental concat era tensors to all_events
    scalar_keys = {"total_product_of_weights", "total_normalization_weights", "total_evaluation_weight"}

    for uid, tensors in era_events.items():

        # when uid not there set base
        if uid not in all_events:
            all_events[uid] = tensors
            continue

        # concatenate data from era to all
        existing = all_events[uid]
        merged = {}
        for key, value in tensors.items():
            if key == "mask":
                merged["mask"] = {
                    mask_key: torch.cat([existing["mask"][mask_key], value[mask_key]], dim=0)
                    for mask_key in value
                }
            elif key in scalar_keys:
                merged[key] = existing[key] + value
            else:
                merged[key] = torch.cat([existing[key], value], dim=0)
        all_events[uid] = merged

    return all_events
