"""
Reader for the parquet files of Marcel's comparison (``.../hbtml_data/mc/<dataset>/data_*_dnn_ext.parquet``).

QUICK AND DIRTY: this exists for the comparison only and is replaced by the column store rework.
The files differ from the root production in three ways, which is why they get their own reader:

    - collections are records (``events.Jet.pt``) instead of flat branches (``Jet_pt``)
    - the regression DNN columns only hold the neutrinos, so the four-vectors of the leptons and the
      b-jet candidates are built from the reconstructed objects (pt, eta, phi, mass)
    - the events are already selected, no cuts are applied

Only derived continuous features (see ``data_handling/derived_features.py``) and bare categorical
columns such as ``channel_id`` can be read. The result has the same layout as the root reader's.
"""
from __future__ import annotations

from typing import Union

import awkward as ak
import numpy as np
import pyarrow.parquet as pq

from bbTT.data_handling.derived_features import FOUR_VECTOR_COMPONENTS, build_derived_features, kinematics
from bbTT.data_handling.eras import JET_COLLECTION, EraSpec
from bbTT.monitoring.logger.logger import get_logger

logger_inst = get_logger(__name__)

# which reconstructed object fills the two lepton slots (vis_tau1, vis_tau2) per channel_id.
# ASSUMPTION: 1 = etau, 2 = mutau, 3 = tautau and the collections hold the selected leptons in order.
LEPTON_SLOTS: dict[int, tuple[tuple[str, int], tuple[str, int]]] = {
    1: (("Electron", 0), ("Tau", 0)),
    2: (("Muon", 0), ("Tau", 0)),
    3: (("Tau", 0), ("Tau", 1)),
}
OBJECT_FIELDS = ("pt", "eta", "phi", "mass")
MET_COLLECTION = "PuppiMET"

# columns every file must have, same meaning as in the root reader
META_FIELDS = ("process_id", "event", "normalization_weight")

# mass windows of the evaluation phase space in GeV, same values as in evaluation_phase_space.py
DI_TAU_MASS_WINDOW = (15.0, 130.0)
DI_BJET_MASS_WINDOW = (40.0, 270.0)


def object_slot(events: ak.Array, collection: str, index: int) -> dict[str, np.ndarray]:
    """
    Kinematics of the *index*-th object of a jagged collection, NaN where the event has no such object.

    Args:
        events (ak.Array): Events holding the collection as record with the fields OBJECT_FIELDS.
        collection (str): Collection name, e.g. "Tau".
        index (int): Position inside the collection.

    Returns:
        dict[str, np.ndarray]: Field name -> float64 values, shape [num_events] each.
    """
    slot = {}
    for field in OBJECT_FIELDS:
        padded = ak.pad_none(events[collection][field], index + 1, axis=1)
        slot[field] = ak.to_numpy(ak.fill_none(padded[:, index], np.nan)).astype(np.float64)
    return slot


def to_cartesian(obj: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Converts (pt, eta, phi, mass) into the four-vector components (e, px, py, pz).

    Args:
        obj (dict[str, np.ndarray]): Fields "pt", "eta", "phi" and "mass".

    Returns:
        dict[str, np.ndarray]: Components "e", "px", "py" and "pz".
    """
    px, py, pz = obj["pt"] * np.cos(obj["phi"]), obj["pt"] * np.sin(obj["phi"]), obj["pt"] * np.sinh(obj["eta"])
    energy = np.sqrt(px**2 + py**2 + pz**2 + obj["mass"] ** 2)
    return {"e": energy, "px": px, "py": py, "pz": pz}


def lepton_slots(events: ak.Array, channel_id: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Picks the two leptons of every event according to its channel, see LEPTON_SLOTS.

    Args:
        events (ak.Array): Events holding the collections Electron, Muon and Tau.
        channel_id (np.ndarray): Channel per event.

    Returns:
        tuple[dict[str, np.ndarray], dict[str, np.ndarray]]: Four-vector components of the first and the
            second lepton. NaN where the expected object does not exist.

    Raises:
        ValueError: If a channel occurs that LEPTON_SLOTS does not know.
    """
    unknown = sorted(set(np.unique(channel_id).tolist()) - set(LEPTON_SLOTS))
    if unknown:
        raise ValueError(f"channel_id values {unknown} have no lepton assignment, known: {sorted(LEPTON_SLOTS)}")

    # read every (collection, index) once, then choose per event
    objects = {slot: object_slot(events, *slot) for slots in LEPTON_SLOTS.values() for slot in slots}
    leptons = []
    for position in (0, 1):
        chosen = {}
        for field in OBJECT_FIELDS:
            values = np.full(len(channel_id), np.nan, dtype=np.float64)
            for channel, slots in LEPTON_SLOTS.items():
                is_channel = channel_id == channel
                values[is_channel] = objects[slots[position]][field][is_channel]
            chosen[field] = values
        leptons.append(to_cartesian(chosen))
    return leptons[0], leptons[1]


def split_branch(branch: str) -> tuple[str, str]:
    """
    Splits a flat branch name of the root production into collection and field.

    Args:
        branch (str): Branch name, e.g. "HHBJet_btagUParTAK4B".

    Returns:
        tuple[str, str]: Collection and field, e.g. ("HHBJet", "btagUParTAK4B").
    """
    collection, field = branch.split("_", 1)
    return collection, field


def mass(source: dict[str, np.ndarray], obj_a: str, obj_b: str) -> np.ndarray:
    """
    Invariant mass of the sum of two objects of *source*.

    Args:
        source (dict[str, np.ndarray]): Four-vector components by registry name, e.g. "bjet1_px".
        obj_a (str): First object, e.g. "bjet1".
        obj_b (str): Second object, e.g. "bjet2".

    Returns:
        np.ndarray: Mass per event in GeV.
    """
    summed = np.stack([source[f"{obj_a}_{c}"] + source[f"{obj_b}_{c}"] for c in FOUR_VECTOR_COMPONENTS], axis=1)
    return kinematics(summed)[3]


def to_struct(columns: dict[str, np.ndarray]) -> np.ndarray:
    """
    Packs flat columns into one structured array, the layout the rest of the pipeline expects.

    Args:
        columns (dict[str, np.ndarray]): Column name -> values, all of the same length.

    Returns:
        np.ndarray: Structured array with one field per column.
    """
    length = len(next(iter(columns.values())))
    struct = np.empty(length, dtype=[(name, values.dtype) for name, values in columns.items()])
    for name, values in columns.items():
        struct[name] = values
    return struct


def load_parquet_and_convert_to_numpy(
    files_path: Union[list[str], str],
    branches: list[str],
    era: EraSpec,
    derived: tuple[str, ...],
    derived_sources: dict[str, str],
    row_masks: tuple[str, ...] = (),
) -> tuple[list[np.ndarray], list[tuple[str, int, int]]]:
    """
    Load all parquet files in *files_path*, build the derived features and return one structured array per file.

    Args:
        files_path (list[str], str): Parquet files or a single file.
        branches (list[str]): Columns the result must contain: derived features and bare columns.
        era (EraSpec): Era of the files, the path does not tell it. Provides the tagger and its working points.
        derived (tuple[str, ...]): Names in *branches* that are derived features. Every continuous feature must
            be one, the files have no regression DNN four-vectors to read them from.
        derived_sources (dict[str, str]): Registry name -> array column of the derived feature sources. Only the
            neutrino entries are read as columns, all other sources are built from the reconstructed objects.
        row_masks (tuple[str, ...], optional): Boolean columns an event must have set to be kept,
            e.g. ("keep_in_union",). Defaults to ().

    Returns:
        tuple[list[np.ndarray], list[tuple[str, int, int]]]: The arrays, and per file the path, the number of
            rows in the file and the number of rows kept.

    Raises:
        ValueError: If a channel has no lepton assignment (see LEPTON_SLOTS).
    """
    if isinstance(files_path, str):
        files_path = [files_path]

    bare_columns = [b for b in branches if b not in derived]
    neutrino_columns = {name: column for name, column in derived_sources.items() if name.startswith("nu")}
    btag_collection, btag_field = split_branch(era.btag_score_branch)
    medium_wp = era.working_point("medium")

    object_columns = [f"{c}.{f}" for c in ("Electron", "Muon", "Tau", btag_collection) for f in OBJECT_FIELDS]
    object_columns += [f"{btag_collection}.{btag_field}", f"{JET_COLLECTION}.pt", f"{JET_COLLECTION}.{btag_field}"]
    object_columns += [f"{MET_COLLECTION}.pt", f"{MET_COLLECTION}.phi"]

    arrays, num_events = [], []
    for current_file, file_path in enumerate(files_path, start=1):
        logger_inst.debug_progress(f"loading file: {current_file}/{len(files_path)}")

        # weights are dataset specific, only those the file has are multiplied
        fields_in_file = set(pq.read_schema(file_path).names)
        weights_in_file = sorted(fields_in_file.intersection(era.transfer_weights))
        read_columns = set(bare_columns).union(META_FIELDS, weights_in_file, neutrino_columns.values(), row_masks, {"channel_id"})
        events = ak.from_parquet(file_path, columns=sorted(read_columns) + object_columns)
        rows_in_file = len(events)

        # --- row selection: the files are pre-selected, only explicitly requested flags are applied ---
        for row_mask in row_masks:
            events = events[ak.to_numpy(events[row_mask]).astype(bool)]

        channel_id = ak.to_numpy(events["channel_id"])

        # --- sources of the derived features, by registry name ---
        source: dict[str, np.ndarray] = {}
        lepton_1, lepton_2 = lepton_slots(events, channel_id)
        bjet_1 = to_cartesian(object_slot(events, btag_collection, 0))
        bjet_2 = to_cartesian(object_slot(events, btag_collection, 1))
        for name, vector in (("vis_tau1", lepton_1), ("vis_tau2", lepton_2), ("bjet1", bjet_1), ("bjet2", bjet_2)):
            for component in FOUR_VECTOR_COMPONENTS:
                source[f"{name}_{component}"] = vector[component]
        # a missing b-jet candidate is a four-vector of zeros, as in the root production
        for name in ("bjet1", "bjet2"):
            for component in FOUR_VECTOR_COMPONENTS:
                source[f"{name}_{component}"] = np.nan_to_num(source[f"{name}_{component}"], nan=0.0)

        for name, column in neutrino_columns.items():
            source[name] = ak.to_numpy(events[column]).astype(np.float64)

        met_pt = ak.to_numpy(events[MET_COLLECTION]["pt"]).astype(np.float64)
        met_phi = ak.to_numpy(events[MET_COLLECTION]["phi"]).astype(np.float64)
        source["met_px"], source["met_py"] = met_pt * np.cos(met_phi), met_pt * np.sin(met_phi)

        # --- jagged collections reduced to one number per event ---
        candidate_scores, jets = events[btag_collection][btag_field], events[JET_COLLECTION]
        reductions = {
            "n_btags_hhbjet": ak.to_numpy(ak.sum(candidate_scores >= medium_wp, axis=1)),
            "ht": ak.to_numpy(ak.sum(jets["pt"], axis=1)),
            "n_jets": ak.to_numpy(ak.num(jets["pt"], axis=1)),
            "n_btags": ak.to_numpy(ak.sum(jets[btag_field] >= medium_wp, axis=1)),
        }
        # the jet features assume that the candidates are part of the jet collection
        fewer_jets = int(np.sum(reductions["n_jets"] < ak.to_numpy(ak.num(candidate_scores, axis=1))))
        if fewer_jets:
            logger_inst.warning(
                f"{fewer_jets} events of {file_path} have fewer jets than HH b-jet candidates: the candidates "
                f"seem not to be part of {JET_COLLECTION}, so log_ht, n_jets and n_btags miss them"
            )

        derived_features = build_derived_features(source, reductions=reductions)
        columns: dict[str, np.ndarray] = {name: derived_features[name] for name in derived}
        for column in bare_columns:
            columns[column] = ak.to_numpy(events[column])

        # --- meta fields and weights ---
        for field in META_FIELDS:
            columns[field] = ak.to_numpy(events[field])
        combined_weight = columns["normalization_weight"].astype(np.float64)
        for weight in weights_in_file:
            # some weights are optional per event, a missing one does not change the event weight
            combined_weight = combined_weight * ak.to_numpy(ak.fill_none(events[weight], 1.0)).astype(np.float64)
        columns["combined_weight"] = combined_weight.astype(np.float32)

        # --- evaluation phase space masks, same definition as for the root production ---
        num_tagged = ak.to_numpy(ak.sum(candidate_scores > medium_wp, axis=1))
        di_tau_mass, di_bjet_mass = mass(source, "vis_tau1", "vis_tau2"), mass(source, "bjet1", "bjet2")
        columns["at_least_1_bjet_mask"] = num_tagged >= 1
        columns["at_least_2_bjet_mask"] = num_tagged >= 2
        columns["di_tau_mask"] = (di_tau_mass >= DI_TAU_MASS_WINDOW[0]) & (di_tau_mass <= DI_TAU_MASS_WINDOW[1])
        columns["di_bjet_mask"] = (di_bjet_mass >= DI_BJET_MASS_WINDOW[0]) & (di_bjet_mass <= DI_BJET_MASS_WINDOW[1])

        events_np = to_struct(columns)
        num_events.append((file_path, rows_in_file, len(events_np)))
        arrays.append(events_np)
    return arrays, num_events
