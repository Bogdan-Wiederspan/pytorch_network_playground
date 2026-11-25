"""
Constants describing the data-taking eras the pipeline knows.

This is deliberately not a config: the values describe how the data lies on disk
(branch names, working points, luminosity) and change only when a new era or a new
production arrives. Nothing in here is ever stored in a checkpoint. Everything that
differs between eras should be looked up from ``ERAS`` instead of being branched on
in the pipeline code.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# column prefix of the regression DNN features per production; the production is the last
# directory of INPUT_DATA_DIR (e.g. ".../prod28"), productions not listed use the default
DEFAULT_COLUMN_PREFIX = "reg_dnn_moe"
PRODUCTION_COLUMN_PREFIX: dict[str, str] = {
    "prod14": "res_dnn_pnet",
    "prod19": "res_dnn_pnet",
}


def column_prefix(production: str) -> str:
    """
    Column prefix of the regression DNN features of a production.

    Args:
        production (str): Production name, e.g. "prod28".

    Returns:
        str: Prefix such as "reg_dnn_moe".
    """
    return PRODUCTION_COLUMN_PREFIX.get(production, DEFAULT_COLUMN_PREFIX)




# collection of all selected jets of an event, the HH b-jet candidates are a subset of it.
# NOTE: the production does not ship it yet, the names follow the HHBJet branches and are an assumption
JET_COLLECTION = "Jet"
JET_PT_BRANCH = f"{JET_COLLECTION}_pt"


@dataclass(frozen=True)
class EraSpec:
    """
    Everything the pipeline needs to know about one data-taking era.

    Args:
        name (str): Era name as used in the data directory, e.g. "22pre".
        luminosity (float): Integrated luminosity of the era in fb^-1.
        btagger (str): Name of the b-tagger the era was produced with. Not a choice: each era has exactly
            one tagger, and eras with different taggers cannot be trained together since their tag scores
            and the derived features are not comparable.
        btag_score_branch (str): Branch holding the per-jet b-tag score of the HH b-jet candidates.
        transfer_weights (frozenset[str]): Weight branches whose product transfers from the training to the
            evaluation phase space. Branches a dataset does not have (e.g. muon weights in dy) are skipped
            when reading, so the set lists everything the era's production can provide.

        btag_working_points (dict[str, float]): Score threshold per working point, e.g. {"medium": 0.245}.
    """
    name: str
    luminosity: float
    btagger: str
    btag_score_branch: str
    transfer_weights: frozenset[str]
    btag_working_points: dict[str, float] = field(default_factory=dict)

    @property
    def jet_btag_score_branch(self) -> str:
        """
        Branch holding the b-tag score of all jets, same tagger as ``btag_score_branch``
        but for the collection JET_COLLECTION instead of the HH b-jet candidates.

        Returns:
            str: Branch name, e.g. "Jet_btagUParTAK4B".
        """
        _, score_name = self.btag_score_branch.split("_", 1)
        return f"{JET_COLLECTION}_{score_name}"

    def working_point(self, wp_level: str) -> float:
        """
        Score threshold of the given working point.

        Args:
            wp_level (str): Working point name, e.g. "medium".

        Returns:
            float: Score threshold.

        Raises:
            ValueError: If the era has no threshold for the working point, e.g. not filled in yet.
        """
        if wp_level not in self.btag_working_points:
            raise ValueError(
                f"Era {self.name} has no {self.btagger} working point '{wp_level}', "
                f"known: {sorted(self.btag_working_points)}"
            )
        return self.btag_working_points[wp_level]

# weights that are valid for eras before 24
PRE_ERA24_WEIGHTS = frozenset({
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
    "top_pt_weight",
})

# weights that are valid for era 24; the b-tag scale factors use a different method than
# the pnet eras and come as "btag_weight", the production has no dy_weight
ERA_24_WEIGHTS = frozenset({
    "btag_weight",
    "normalized_pdf_weight",
    "normalized_murmuf_weight",
    "normalized_pu_weight",
    "normalized_isr_weight",
    "normalized_fsr_weight",
    "electron_id_weight",
    "electron_reco_weight",
    "muon_id_weight",
    "muon_iso_weight",
    "tau_weight",
    "trigger_weight",
    "top_pt_weight",
})


# ParticleNet (22/23): branch names are shared, only the working points differ per era
# working points as taken from https://github.com/uhh-cms/hh2bbtautau/blob/master/hbt/config/configs_hbt.py#L1252
_PNET_SCORE_BRANCH = "HHBJet_btagPNetB"

# UParT (24): ParticleNet is not calibrated for 24, the transformer-based UParT is used instead
_UPART_SCORE_BRANCH = "HHBJet_btagUParTAK4B"

# the insertion order is the chronological order of the eras, it fixes the merge order of the
# era caches (see DataConfig.ordered_eras); DataConfig keeps the matching ERAS_CHOICE literal as the
# visible list of what can be configured and checks that both agree
ERAS: dict[str, EraSpec] = {
    "22pre": EraSpec(
        name="22pre",
        luminosity=7.9,
        btagger="pnet",
        btag_score_branch=_PNET_SCORE_BRANCH,
        transfer_weights=PRE_ERA24_WEIGHTS,
        btag_working_points={"loose": 0.047, "medium": 0.245, "tight": 0.6734, "xtight": 0.7862, "xxtight": 0.961},
    ),
    "22post": EraSpec(
        name="22post",
        luminosity=26.6,
        btagger="pnet",
        btag_score_branch=_PNET_SCORE_BRANCH,
        transfer_weights=PRE_ERA24_WEIGHTS,
        btag_working_points={"loose": 0.0499, "medium": 0.2605, "tight": 0.6915, "xtight": 0.8033, "xxtight": 0.9664},
    ),
    "23pre": EraSpec(
        name="23pre",
        luminosity=18,
        btagger="pnet",
        btag_score_branch=_PNET_SCORE_BRANCH,
        transfer_weights=PRE_ERA24_WEIGHTS,
        btag_working_points={"loose": 0.0358, "medium": 0.1917, "tight": 0.6172, "xtight": 0.7515, "xxtight": 0.9659},
    ),
    "23post": EraSpec(
        name="23post",
        luminosity=9.6,
        btagger="pnet",
        btag_score_branch=_PNET_SCORE_BRANCH,
        transfer_weights=PRE_ERA24_WEIGHTS,
        btag_working_points={"loose": 0.0359, "medium": 0.1919, "tight": 0.6133, "xtight": 0.7544, "xxtight": 0.9688},
    ),
    "24": EraSpec(
        name="24",
        luminosity=110,
        btagger="upart",
        btag_score_branch=_UPART_SCORE_BRANCH,
        transfer_weights=ERA_24_WEIGHTS,
        btag_working_points={"loose": 0.0246, "medium": 0.1271, "tight": 0.4648, "xtight": 0.6298, "xxtight": 0.9739},
    ),
}

def era_order(era: str) -> int:
    """
    Chronological position of an era, used to sort era sets independent of how they are written.

    Args:
        era (str): Era name.

    Returns:
        int: Position in ``ERAS``.
    """
    return tuple(ERAS).index(era)
