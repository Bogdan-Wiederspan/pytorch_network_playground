import awkward as ak


def particle_net_wp(year, wp_level="medium"):
    # as taken from https://github.com/uhh-cms/hh2bbtautau/blob/master/hbt/config/configs_hbt.py#L1252
    particle_net_wp = {
        "loose": {"22pre": 0.047, "22post": 0.0499, "23pre": 0.0358, "23post": 0.0359, "2024": None}[year],
        "medium": {"22pre": 0.245, "22post": 0.2605, "23pre": 0.1917, "23post": 0.1919, "2024": None}[year],
        "tight": {"22pre": 0.6734, "22post": 0.6915, "23pre": 0.6172, "23post": 0.6133, "2024": None}[year],
        "xtight": {"22pre": 0.7862, "22post": 0.8033, "23pre": 0.7515, "23post": 0.7544, "2024": None}[year],
        "xxtight": {"22pre": 0.961, "22post": 0.9664, "23pre": 0.9659, "23post": 0.9688, "2024": None}[year],
    }
    return particle_net_wp[wp_level]


def di_tau_mass_window(events, suffix="res_dnn_pnet"):
    """Reconstruct Lepton Pair Mass for given *events*, which are saved with given *suffix* """
    # tau mass window
    l_px = events[f"{suffix}_vis_tau1_px"] + events[f"{suffix}_vis_tau2_px"]
    l_py = events[f"{suffix}_vis_tau1_py"] + events[f"{suffix}_vis_tau2_py"]
    l_pz = events[f"{suffix}_vis_tau1_pz"] + events[f"{suffix}_vis_tau2_pz"]
    l_e = events[f"{suffix}_vis_tau1_e"] + events[f"{suffix}_vis_tau2_e"]

    # since no coffee behavior, calculate mass by manually from 4 vector
    di_tau_mass = (l_e**2 - (l_px**2 + l_py**2 + l_pz**2)) ** 0.5
    di_tau_mass_window_mask = (di_tau_mass >= 15) & (di_tau_mass <= 130)
    return di_tau_mass_window_mask


def b_jet_mask(events, year="22pre", wp_level="medium", atleast_num=1):
    """Select events with *atleast_num* of b-tagges values according to ParticleNet score of given *year* and *wp_level*"""
    b_tag_wp = particle_net_wp(year, wp_level=wp_level)
    return ak.sum(events.HHBJet_btagPNetB > b_tag_wp, axis=1) >= atleast_num


def di_b_jet_mask_window(events, suffix="res_dnn_pnet"):
    """Reconstruct b-Jet Pair Mass for given *events*, which are saved with given *suffix* """
    b_px = events[f"{suffix}_bjet1_px"] + events[f"{suffix}_bjet2_px"]
    b_py = events[f"{suffix}_bjet1_py"] + events[f"{suffix}_bjet2_py"]
    b_pz = events[f"{suffix}_bjet1_pz"] + events[f"{suffix}_bjet2_pz"]
    b_e = events[f"{suffix}_bjet1_e"] + events[f"{suffix}_bjet2_e"]
    di_bjet_mass = (b_e**2 - (b_px**2 + b_py**2 + b_pz**2)) ** 0.5

    di_bjet_mass_window_mask = (di_bjet_mass >= 40) & (di_bjet_mass <= 270)
    return di_bjet_mass_window_mask


def res1b_and_res2b_phase_space_mask(events: ak.Array, year: list[str], suffix: str = "res_dnn_pnet") -> tuple[ak.Array, ak.Array, ak.Array]:
    """
    Calculates Masks to get into our evaluation phase space.
    Depends on *year*, apply  base *cut* and depending on the producer add a *suffix* to fields in root file.
    Definition of mask is defined in https://github.com/uhh-cms/hh2bbtautau/blob/master/hbt/categorization/default.py#L206-L240

    Args:
        events (ak.Array): Awkward Array with necessary events.
        year (list[str]): Year string e.g. "22pre"
        suffix (str, optional): Suffix for fields in uproot file. Defaults to "res_dnn_pnet".

    Returns:
        ak.array: Masks for events within lepton pair and b-tagged pair, aswell for bjets
    """
    di_tau_mass_window_mask = di_tau_mass_window(events, suffix=suffix)
    bjet_mask = b_jet_mask(events, year=year, wp_level="medium", atleast_num=1)
    di_bjet_mass_window_mask = di_b_jet_mask_window(events, suffix=suffix)
    return di_tau_mass_window_mask, di_bjet_mass_window_mask, bjet_mask
