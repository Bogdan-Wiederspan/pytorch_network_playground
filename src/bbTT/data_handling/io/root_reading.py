import pathlib
from typing import Union

import awkward as ak
import uproot

from bbTT.data_handling.eras import ERAS
from bbTT.data_handling.evaluation_phase_space import res1b_and_res2b_phase_space_mask
from bbTT.monitoring.logger.logger import get_logger

logger_inst = get_logger(__name__)

def load_root_and_convert_to_numpy(
    files_path: Union[list[str], str],
    branches: Union[list[str], str, None] = None,
    cut: list[str] = None,
    jagged: Union[dict[str, tuple[str, int, float]], None] = None,
) -> ak.Array:
    """
    Load all root files in *files_path* and return them as a single awkward array.
    If only certain branches are needed, they can be specified in *branches*.
    To prevent loading unnecessary data a list of *cut*s can be added.

    Args:
        files_path (list[str], str): list of root files or single root file
        branches (list[str], str, optional): branches that should be loaded e.g. ["events", "run"]. If None loads all branches . Defaults to None.
        jagged (dict[str, tuple[str, int, float]], optional): Flat column -> (branch, index, fill): the column is
            created from entry *index* of the jagged *branch*, *fill* is used when the entry is missing.
            Such columns may be listed in *branches*, the branch itself is read instead. Defaults to None.

    Returns:
        ak.Array: awkward array containing all data from the root files

    """
    # set of branches that are always extracted from root files
    meta_fields = {
        "process_id",  # filtering of sub-phase spaces
        "event",  # event number used for k-fold splitting
        "normalization_weight",  # oversampling weight that defines the fraction within batch
    }

    # columns necessary for evaluation masks, the era specific b-tag score is added per file below
    suffix = "res_dnn_pnet" if branches[0].startswith("res_dnn_pnet") else "reg_dnn_moe"
    mask_fields = (
        [f"{suffix}_vis_tau{num}_{kin}" for num in ("1", "2") for kin in ("px", "py", "pz", "e")]
        + ["HHBJet_mass"]
        + [f"{suffix}_bjet{num}_{kin}" for num in ("1", "2") for kin in ("px", "py", "pz", "e")]
    )

    # training and evaluation phase space are not the same
    # a transfer weight can be calculated using the product of these weights
    # the b-tag weight depends on the era's tagger and is added per file below

    # combine all branches to have only 1 uproot -> ak -> numpy chain;
    # columns split from jagged branches are not in the tree, their source branch is read instead
    jagged = jagged or {}
    jagged_sources = {branch for branch, _, _ in jagged.values()}
    all_branches = (set(branches) - set(jagged)).union(meta_fields, mask_fields, jagged_sources)

    # handle edge cases for cuts and file paths
    if isinstance(cut, str):
        cut = [cut]
    cut = "&".join(cut) if cut else None
    if isinstance(files_path, str):
        files_path = [files_path]

    # conversion
    arrays = []
    num_events = []
    max_files = len(files_path)

    for current_file, file_path in enumerate(files_path, start=1):
        logger_inst.debug_progress(f"loading file: {current_file}/{max_files}")

        # path is ${INPUT_DATA_DIR}/ERA/dataset/file.root, the era decides tagger branches and working points
        era = ERAS[pathlib.Path(file_path).parents[1].stem]

        with uproot.open(file_path, object_cache=None, array_cache=None) as file:
            tree = file["events"]
            # some weights are dataset specific and needs to be extracted on uid base
            # e.x. muon_id_weight does not exist in dy datasets
            weights_in_current_file = set(tree.keys()).intersection(era.transfer_weights)
            read_fields = all_branches.union(weights_in_current_file, {era.btag_score_branch})

            events = tree.arrays(read_fields, library="ak", cut=cut)

            # --- jagged splitting ---
            # split jagged branches into flat per-jet columns, e.g. the UParT score of the first and second b-jet
            for column, (branch, index, fill) in jagged.items():
                padded = ak.pad_none(events[branch], index + 1, axis=1)
                events[column] = ak.fill_none(padded[:, index], fill)

            # calculate_monte_carlo_weight
            # this weight combines all corrections, but also over-sampling of mc generator
            combined_weight = events["normalization_weight"]

            for w in weights_in_current_file:
                combined_weight = combined_weight * events[w]
            events["combined_weight"] = combined_weight

            # calculate evaluation phase space mask
            di_tau_mask, di_bjet_mask, at_least1_btag_mask, at_least2_btag_mask = res1b_and_res2b_phase_space_mask(
                events=events, era=era, suffix=suffix
            )

            events["at_least_1_bjet_mask"] = at_least1_btag_mask
            events["at_least_2_bjet_mask"] = at_least2_btag_mask
            events["di_tau_mask"] = di_tau_mask
            events["di_bjet_mask"] = di_bjet_mask

            # drop inputs, and keep only artifacts and results
            keep = (
                set(branches).union(meta_fields).union({"combined_weight", "at_least_1_bjet_mask", "at_least_2_bjet_mask", "di_tau_mask", "di_bjet_mask"})
            )

            events_np = events[list(keep)].to_numpy()
            del events

            num_events.append((file_path, tree.num_entries, len(events_np)))
            arrays.append(events_np)
    return arrays, num_events
