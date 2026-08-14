import awkward as ak


def res1b_and_res2b_phase_space_mask(events: ak.Array, year: list[str], suffix: str="res_dnn_pnet"):
    """
    Calculates Masks to get into our evaluation phase space.
    Depends on *year*, apply  base *cut* and depending on the producer add a *suffix* to fields in root file.
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
