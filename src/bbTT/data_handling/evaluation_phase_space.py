from __future__ import annotations

import awkward as ak

from bbTT.data_handling.eras import EraSpec


def di_tau_mass_window(events, suffix="res_dnn_pnet"):
    """Reconstruct Lepton Pair Mass for given *events*, which are saved with given *suffix*"""
    # tau mass window
    l_px = events[f"{suffix}_vis_tau1_px"] + events[f"{suffix}_vis_tau2_px"]
    l_py = events[f"{suffix}_vis_tau1_py"] + events[f"{suffix}_vis_tau2_py"]
    l_pz = events[f"{suffix}_vis_tau1_pz"] + events[f"{suffix}_vis_tau2_pz"]
    l_e = events[f"{suffix}_vis_tau1_e"] + events[f"{suffix}_vis_tau2_e"]

    # since no coffee behavior, calculate mass by manually from 4 vector
    di_tau_mass = (l_e**2 - (l_px**2 + l_py**2 + l_pz**2)) ** 0.5
    di_tau_mass_window_mask = (di_tau_mass >= 15) & (di_tau_mass <= 130)
    return di_tau_mass_window_mask


def b_jet_mask(events: ak.Array, era: EraSpec, wp_level: str = "medium", atleast_num: int = 1) -> ak.Array:
    """
    Select events with at least *atleast_num* b-tagged jets. Which tagger, which branch and which
    threshold apply is taken from the era constants, so this function knows nothing about ParticleNet or UParT.

    Args:
        events (ak.Array): Events holding the era's b-tag score branch.
        era (EraSpec): Era the events belong to.
        wp_level (str, optional): Working point name. Defaults to "medium".
        atleast_num (int, optional): Minimum number of tagged jets. Defaults to 1.

    Returns:
        ak.Array: Boolean mask per event.
    """
    b_tag_wp = era.working_point(wp_level)
    return ak.sum(events[era.btag_score_branch] > b_tag_wp, axis=1) >= atleast_num


def di_b_jet_mask_window(events, suffix="res_dnn_pnet"):
    """Reconstruct b-Jet Pair Mass for given *events*, which are saved with given *suffix*"""
    b_px = events[f"{suffix}_bjet1_px"] + events[f"{suffix}_bjet2_px"]
    b_py = events[f"{suffix}_bjet1_py"] + events[f"{suffix}_bjet2_py"]
    b_pz = events[f"{suffix}_bjet1_pz"] + events[f"{suffix}_bjet2_pz"]
    b_e = events[f"{suffix}_bjet1_e"] + events[f"{suffix}_bjet2_e"]
    di_bjet_mass = (b_e**2 - (b_px**2 + b_py**2 + b_pz**2)) ** 0.5

    di_bjet_mass_window_mask = (di_bjet_mass >= 40) & (di_bjet_mass <= 270)
    return di_bjet_mass_window_mask


def res1b_and_res2b_phase_space_mask(events: ak.Array, era: EraSpec, suffix: str = "res_dnn_pnet") -> tuple[ak.Array, ak.Array, ak.Array]: # noqa
    """
    Calculates Masks to get into our evaluation phase space.
    Depends on *era*, apply  base *cut* and depending on the producer add a *suffix* to fields in root file.
    Definition of mask is defined in https://github.com/uhh-cms/hh2bbtautau/blob/master/hbt/categorization/default.py#L206-L240

    Args:
        events (ak.Array): Awkward Array with necessary events.
        era (EraSpec): Era the events belong to, provides the b-tag branch and working points.
        suffix (str, optional): Suffix for fields in uproot file. Defaults to "res_dnn_pnet".

    Returns:
        ak.array: Masks for events within lepton pair and b-tagged pair, aswell for bjets
    """
    di_tau_mass_window_mask = di_tau_mass_window(events, suffix=suffix)
    at_least_one_btag_mask = b_jet_mask(events, era=era, wp_level="medium", atleast_num=1)
    at_least_two_btag_mask = b_jet_mask(events, era=era, wp_level="medium", atleast_num=2)

    di_bjet_mass_window_mask = di_b_jet_mask_window(events, suffix=suffix)
    return di_tau_mass_window_mask, di_bjet_mass_window_mask, at_least_one_btag_mask, at_least_two_btag_mask
