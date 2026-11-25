"""
Features that are not columns of the production but are calculated from them when the data cache is built.

The set follows the high level features of Marcel's comparison network (his "global_cont" feature set):
masses, transverse momenta and angular distances of the composite systems instead of raw four-vectors.
The calculation is plain numpy and knows nothing about root files, the reader hands in the source
columns and stores the result like any other column.

Naming of the systems:
    l1, l2  visible tau decay products (``vis_tau1``, ``vis_tau2``)
    b1, b2  b-jet candidates (``bjet1``, ``bjet2``)
    ll, bb  sum of the two leptons / the two b-jets
    bbll    bb + ll
    tt      di-tau system: leptons plus regressed neutrinos (``nu1``, ``nu2``)
    hh      bb + tt

Three features need the collection of all jets (JET_FEATURES). It is jagged, so the reader reduces it
to one number per event and hands the result in as *reductions*. As long as the production does not
ship that collection these features cannot be built; ``n_btags_hhbjet`` is the stand-in for ``n_btags``
that only counts the two HH b-jet candidates.
"""
from __future__ import annotations

import numpy as np

from bbTT.utils.constants import EMPTY_FLOAT

# numerical guards, same values as in the reference implementation
EPS = 1e-6
LOG_FLOOR = 1e-3
HIGGS_MASS = 125.0

# marker written where a feature cannot be built because a b-jet candidate is missing.
# It must be a real number: events with NaN in a feature are dropped when the cache is built.
MISSING = EMPTY_FLOAT

FOUR_VECTOR_COMPONENTS = ("e", "px", "py", "pz")

# registry names of the columns the calculation reads, the reader resolves them to array columns
SOURCE_FEATURES: tuple[str, ...] = (
    *(f"{obj}_{c}" for obj in ("vis_tau1", "vis_tau2", "bjet1", "bjet2") for c in FOUR_VECTOR_COMPONENTS),
    *(f"{obj}_{c}" for obj in ("nu1", "nu2") for c in ("px", "py", "pz")),
    "met_px",
    "met_py",
)

# features that are always defined
ALWAYS_PRESENT: tuple[str, ...] = (
    "log_met_pt",
    "log_ll_mass",
    "dr_l1_l2",
    "log_ll_pt",
    "dphi_ll_met",
    "log1p_mt_l1_met",
    "log_tt_mass",
    "log_tt_pt",
    "x1",
    "x2",
)

# features that need b-jet candidates: name -> candidates that must exist (1 = bjet1, 2 = bjet2)
NEEDS_BJETS: dict[str, tuple[int, ...]] = {
    "log_bb_mass": (1, 2),
    "log_bb_pt": (1, 2),
    "dr_b1_b2": (1, 2),
    "deta_bb_ll": (1, 2),
    "dr_bb_ll": (1, 2),
    "cos_bb_ll": (1, 2),
    "log_bbll_mass": (1, 2),
    "log_bbll_pt": (1, 2),
    "bbll_eta": (1, 2),
    "dr_l1_b1": (1,),
    "dr_l1_b2": (2,),
    "dr_l2_b1": (1,),
    "dr_l2_b2": (2,),
    "dr_bb_tt": (1, 2),
    "log_hh_mass": (1, 2),
    "log_hh_mass_red": (1, 2),
    "log_hh_pt": (1, 2),
    "cos_theta_star": (1, 2),
}

# features from the collection of all jets: name -> reduction of that collection the reader must provide
JET_FEATURES: dict[str, str] = {
    "log_ht": "ht",  # scalar sum of the jet pt
    "n_jets": "n_jets",  # number of jets
    "n_btags": "n_btags",  # number of jets above the medium working point of the era's tagger
}

# features from the HH b-jet candidates only: name -> reduction of the candidates the reader must provide
HHBJET_FEATURES: dict[str, str] = {
    "n_btags_hhbjet": "n_btags_hhbjet",  # number of candidates above the medium working point, at most 2
}

# every feature build_derived_features can return
DERIVED_FEATURES: tuple[str, ...] = ALWAYS_PRESENT + tuple(NEEDS_BJETS) + tuple(JET_FEATURES) + tuple(HHBJET_FEATURES)


def kinematics(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Transverse momentum, pseudorapidity, azimuth and mass of four-vectors.

    Args:
        vector (np.ndarray): Four-vectors with the components (e, px, py, pz), shape [num_events, 4].

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: pt, eta, phi and mass, shape [num_events] each.
            The mass of a space-like vector is 0.
    """
    energy, px, py, pz = vector[:, 0], vector[:, 1], vector[:, 2], vector[:, 3]
    pt = np.sqrt(px**2 + py**2)
    # eta = arcsinh(pz / pt), the floor keeps a vector along the beam axis finite
    eta = np.arcsinh(pz / np.maximum(pt, EPS))
    phi = np.arctan2(py, px)
    mass = np.sqrt(np.maximum(energy**2 - (px**2 + py**2 + pz**2), 0.0))
    return pt, eta, phi, mass


def wrap_phi(delta_phi: np.ndarray) -> np.ndarray:
    """
    Maps an azimuth difference into [-pi, pi).

    Args:
        delta_phi (np.ndarray): Difference of two azimuth angles.

    Returns:
        np.ndarray: The same angle in [-pi, pi).
    """
    return (delta_phi + np.pi) % (2 * np.pi) - np.pi


def delta_r(eta_a: np.ndarray, phi_a: np.ndarray, eta_b: np.ndarray, phi_b: np.ndarray) -> np.ndarray:
    """
    Angular distance $\\Delta R = \\sqrt{\\Delta\\eta^2 + \\Delta\\phi^2}$ of two directions.

    Args:
        eta_a (np.ndarray): Pseudorapidity of the first direction.
        phi_a (np.ndarray): Azimuth of the first direction.
        eta_b (np.ndarray): Pseudorapidity of the second direction.
        phi_b (np.ndarray): Azimuth of the second direction.

    Returns:
        np.ndarray: Delta R per event.
    """
    return np.sqrt((eta_a - eta_b) ** 2 + wrap_phi(phi_a - phi_b) ** 2)


def safe_log(x: np.ndarray) -> np.ndarray:
    """
    Logarithm with a floor of LOG_FLOOR, so 0 and negative inputs stay finite.

    Args:
        x (np.ndarray): Values, e.g. a mass in GeV.

    Returns:
        np.ndarray: log(max(x, LOG_FLOOR)).
    """
    return np.log(np.maximum(x, LOG_FLOOR))


def four_vector(source: dict[str, np.ndarray], obj: str) -> np.ndarray:
    """
    Stacks the components of one object into four-vectors.

    Args:
        source (dict[str, np.ndarray]): Source columns by registry name, e.g. "bjet1_px".
        obj (str): Object name, e.g. "bjet1".

    Returns:
        np.ndarray: Four-vectors (e, px, py, pz) as float64, shape [num_events, 4].
    """
    return np.stack([source[f"{obj}_{c}"] for c in FOUR_VECTOR_COMPONENTS], axis=1).astype(np.float64)


def neutrino_four_vector(source: dict[str, np.ndarray], obj: str) -> np.ndarray:
    """
    Four-vectors of a regressed neutrino. The production stores only the momentum, the energy
    follows from the massless neutrino: $E = |\\vec{p}|$.

    Args:
        source (dict[str, np.ndarray]): Source columns by registry name, e.g. "nu1_px".
        obj (str): Object name, "nu1" or "nu2".

    Returns:
        np.ndarray: Four-vectors (e, px, py, pz) as float64, shape [num_events, 4].
    """
    momentum = np.stack([source[f"{obj}_{c}"] for c in ("px", "py", "pz")], axis=1).astype(np.float64)
    energy = np.sqrt((momentum**2).sum(axis=1))
    return np.concatenate([energy[:, None], momentum], axis=1)


def cos_theta_star(system: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """
    $|\\cos\\theta^*|$ of *system* w.r.t. the beam axis after a boost into the rest frame of *frame*.

    Args:
        system (np.ndarray): Four-vectors that are boosted, shape [num_events, 4].
        frame (np.ndarray): Four-vectors defining the rest frame, shape [num_events, 4].

    Returns:
        np.ndarray: Absolute cosine of the polar angle in the rest frame, in [0, 1].
    """
    beta = frame[:, 1:] / np.maximum(frame[:, :1], EPS)
    beta_squared = np.clip((beta**2).sum(axis=1), EPS, 1 - EPS)
    gamma = 1 / np.sqrt(1 - beta_squared)
    beta_dot_p = (beta * system[:, 1:]).sum(axis=1)
    # spatial part of the Lorentz boost: p' = p + ((gamma - 1) * (beta.p) / beta^2 - gamma * E) * beta
    boosted = system[:, 1:] + ((gamma - 1) * beta_dot_p / beta_squared - gamma * system[:, 0])[:, None] * beta
    return np.abs(boosted[:, 2] / np.maximum(np.linalg.norm(boosted, axis=1), EPS))


def build_derived_features(
    source: dict[str, np.ndarray],
    reductions: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Calculates the derived features of DERIVED_FEATURES.

    A b-jet candidate is missing when the production wrote its four-vector as 0. Every feature that
    needs a missing candidate (see NEEDS_BJETS) is set to MISSING for that event, so the statistics
    can exclude it and the padding layer can replace it.

    The features of JET_FEATURES and HHBJET_FEATURES are only returned when their reduction is in
    *reductions*, all others always.

    Args:
        source (dict[str, np.ndarray]): Columns of SOURCE_FEATURES by registry name, shape [num_events] each.
        reductions (dict[str, np.ndarray]): Jagged collections reduced to one number per event, keyed as in
            JET_FEATURES and HHBJET_FEATURES (e.g. "ht", "n_jets"), shape [num_events] each. They depend on
            jagged branches and on the tagger of the era, so the reader builds them.

    Returns:
        dict[str, np.ndarray]: Feature name -> float32 values, shape [num_events] each.

    Raises:
        KeyError: If a column of SOURCE_FEATURES is not in *source*.
    """
    lepton_1, lepton_2 = four_vector(source, "vis_tau1"), four_vector(source, "vis_tau2")
    bjet_1, bjet_2 = four_vector(source, "bjet1"), four_vector(source, "bjet2")
    tau_1 = lepton_1 + neutrino_four_vector(source, "nu1")
    tau_2 = lepton_2 + neutrino_four_vector(source, "nu2")

    met_px, met_py = source["met_px"].astype(np.float64), source["met_py"].astype(np.float64)
    met_pt, met_phi = np.sqrt(met_px**2 + met_py**2), np.arctan2(met_py, met_px)

    # composite systems
    ll, bb, tt = lepton_1 + lepton_2, bjet_1 + bjet_2, tau_1 + tau_2
    bbll, hh = bb + ll, bb + tt

    l1_pt, l1_eta, l1_phi, _ = kinematics(lepton_1)
    _, l2_eta, l2_phi, _ = kinematics(lepton_2)
    _, b1_eta, b1_phi, _ = kinematics(bjet_1)
    _, b2_eta, b2_phi, _ = kinematics(bjet_2)
    ll_pt, ll_eta, ll_phi, ll_mass = kinematics(ll)
    bb_pt, bb_eta, bb_phi, bb_mass = kinematics(bb)
    bbll_pt, bbll_eta, _, bbll_mass = kinematics(bbll)
    tt_pt, tt_eta, tt_phi, tt_mass = kinematics(tt)
    hh_pt, _, _, hh_mass = kinematics(hh)

    # cosine of the opening angle between bb and ll
    bb_momentum, ll_momentum = bb[:, 1:], ll[:, 1:]
    momentum_norm = np.sqrt((bb_momentum**2).sum(axis=1) * (ll_momentum**2).sum(axis=1))
    cos_bb_ll = (bb_momentum * ll_momentum).sum(axis=1) / np.maximum(momentum_norm, EPS)

    # transverse mass of l1 and met, with 1 - cos(x) = 2 sin^2(x / 2) for stability at small angles
    mt_l1_met = 2 * np.sqrt(l1_pt * met_pt) * np.abs(np.sin(wrap_phi(l1_phi - met_phi) / 2))

    # hh mass with the resolution of both higgs mass reconstructions removed
    hh_mass_reduced = hh_mass - (bb_mass - HIGGS_MASS) - (tt_mass - HIGGS_MASS)

    features = {
        "log_met_pt": safe_log(met_pt),
        "log_ll_mass": safe_log(ll_mass),
        "dr_l1_l2": delta_r(l1_eta, l1_phi, l2_eta, l2_phi),
        "log_ll_pt": safe_log(ll_pt),
        "dphi_ll_met": wrap_phi(ll_phi - met_phi),
        "log1p_mt_l1_met": np.log1p(np.maximum(mt_l1_met / 10.0, LOG_FLOOR)),
        "log_tt_mass": safe_log(tt_mass),
        "log_tt_pt": safe_log(tt_pt),
        # visible energy fraction of each tau
        "x1": lepton_1[:, 0] / np.maximum(tau_1[:, 0], EPS),
        "x2": lepton_2[:, 0] / np.maximum(tau_2[:, 0], EPS),
        "log_bb_mass": safe_log(bb_mass),
        "log_bb_pt": safe_log(bb_pt),
        "dr_b1_b2": delta_r(b1_eta, b1_phi, b2_eta, b2_phi),
        "deta_bb_ll": bb_eta - ll_eta,
        "dr_bb_ll": delta_r(bb_eta, bb_phi, ll_eta, ll_phi),
        "cos_bb_ll": cos_bb_ll,
        "log_bbll_mass": safe_log(bbll_mass),
        "log_bbll_pt": safe_log(bbll_pt),
        "bbll_eta": bbll_eta,
        "dr_l1_b1": delta_r(l1_eta, l1_phi, b1_eta, b1_phi),
        "dr_l1_b2": delta_r(l1_eta, l1_phi, b2_eta, b2_phi),
        "dr_l2_b1": delta_r(l2_eta, l2_phi, b1_eta, b1_phi),
        "dr_l2_b2": delta_r(l2_eta, l2_phi, b2_eta, b2_phi),
        "dr_bb_tt": delta_r(bb_eta, bb_phi, tt_eta, tt_phi),
        "log_hh_mass": safe_log(hh_mass),
        "log_hh_mass_red": safe_log(hh_mass_reduced),
        "log_hh_pt": safe_log(hh_pt),
        "cos_theta_star": cos_theta_star(system=tt, frame=hh),
    }

    # a candidate the production did not find is a four-vector of zeros
    bjet_missing = {1: np.all(bjet_1 == 0.0, axis=1), 2: np.all(bjet_2 == 0.0, axis=1)}
    for name, needed in NEEDS_BJETS.items():
        missing = np.zeros(len(met_pt), dtype=bool)
        for candidate in needed:
            missing |= bjet_missing[candidate]
        features[name] = np.where(missing, MISSING, features[name])

    # features from reduced jagged collections, only those the reader provided
    if "ht" in reductions:
        features["log_ht"] = safe_log(reductions["ht"].astype(np.float64))
    for name in ("n_jets", "n_btags", "n_btags_hhbjet"):
        if name in reductions:
            features[name] = reductions[name].astype(np.float64)

    return {name: features[name].astype(np.float32) for name in DERIVED_FEATURES if name in features}
