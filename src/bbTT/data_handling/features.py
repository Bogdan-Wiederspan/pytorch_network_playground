from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from bbTT.data_handling.derived_features import ALWAYS_PRESENT, HHBJET_FEATURES, JET_FEATURES, MISSING, NEEDS_BJETS
from bbTT.data_handling.eras import ERAS

FEATURE_KIND = Literal["continuous", "categorical"]


@dataclass(frozen=True)
class FeatureSpec:
    """
    One input column, either as the production provides it or derived from such columns.

    Args:
        name (str): Short name used in configs, e.g. "bjet1_px".
        kind (str): "continuous" (standardization path) or "categorical" (embedding path).
        dummy (float, optional): Marker the upstream pipeline writes when the value is missing,
            e.g. 0.0 for the four-vector of an absent jet. None means the column is never missing.
        values (tuple[int, ...]): Categorical only: every value the column takes in the data.
        value_groups (dict[str, tuple[int, ...]], optional): Categorical only: a physics grouping of the values,
            e.g. {"1prong": (0, 1), "3prong": (10, 11)} for decay modes. Monitoring uses it to sort value pairs
            into "same group" and "cross group". Groups must be disjoint and cover every value.
        column (str, optional): Explicit column name. None means "<production prefix>_<name>".
        jagged (tuple[str, int], optional): (branch, index) when the value is one entry of a jagged
            branch, e.g. the score of the first HH b-jet candidate. The reader splits the branch
            into a flat column named like the feature and fills a missing entry with ``dummy``.
        derived (bool): True when the production has no such column and the reader calculates it from other
            columns while the cache is built, see ``data_handling/derived_features.py``.
        unavailable_in (frozenset[str]): Eras whose production does not provide the column.
    """

    name: str
    kind: FEATURE_KIND
    dummy: Optional[float] = None
    values: tuple[int, ...] = ()
    value_groups: Optional[dict[str, tuple[int, ...]]] = None
    column: Optional[str] = None
    jagged: Optional[tuple[str, int]] = None
    derived: bool = False
    unavailable_in: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.kind == "categorical" and not self.values:
            raise ValueError(f"categorical feature {self.name} needs its values")
        if self.kind == "continuous" and self.values:
            raise ValueError(f"continuous feature {self.name} must not define values")
        if self.value_groups is not None:
            grouped = [v for group in self.value_groups.values() for v in group]
            if self.kind != "categorical":
                raise ValueError(f"feature {self.name} is continuous and cannot define value_groups")
            if len(grouped) != len(set(grouped)):
                raise ValueError(f"value_groups of feature {self.name} overlap")
            if set(grouped) != set(self.values):
                raise ValueError(f"value_groups of feature {self.name} must cover exactly its values {self.values}")

        if self.jagged is not None:
            if self.column is not None:
                raise ValueError(f"feature {self.name} is split from a jagged branch and cannot name a column")
            if self.dummy is None:
                raise ValueError(f"feature {self.name} is split from a jagged branch and needs a dummy to fill missing entries")

        if self.derived and (self.column is not None or self.jagged is not None):
            raise ValueError(f"feature {self.name} is derived and cannot name a column or a jagged branch")

    def column_name(self, prefix: str) -> str:
        """
        Actual array column of this feature for a production.

        Args:
            prefix (str): Column prefix of the production, see ``eras.column_prefix``.

        Returns:
            str: Column name, e.g. "reg_dnn_moe_bjet1_px"; the feature name itself for a jagged split
                or a derived feature.

        """
        # in case of jagged arrays a split happens on the mother column, a derived feature is created by the reader
        if self.jagged is not None or self.derived:
            return self.name
        return self.column if self.column is not None else f"{prefix}_{self.name}"

    def is_available(self, era: str) -> bool:
        """
        Whether the production provides this column for an era.

        Args:
            era (str): Era name, e.g. "24".

        Returns:
            bool: False if the era is listed in ``unavailable_in``.
        """
        return era not in self.unavailable_in


# --- constructors: the upstream conventions, written down once ---

def components(
    obj: str,
    names: tuple[str, ...],
    dummy: Optional[float] = None,
    unavailable_in: frozenset[str] = frozenset(),
) -> tuple[FeatureSpec, ...]:
    """Continuous specs of the columns "<obj>_<component>" of one object."""
    return tuple(
        FeatureSpec(name=f"{obj}_{n}", kind="continuous", dummy=dummy, unavailable_in=unavailable_in)
        for n in names
    )


def _registry(*specs: FeatureSpec) -> dict[str, FeatureSpec]:
    """
    Build the name -> spec table and reject duplicate names.

    Args:
        *specs (FeatureSpec): The specs, in definition order.

    Returns:
        dict[str, FeatureSpec]: Registry in definition order.
    """
    registry: dict[str, FeatureSpec] = {}
    for spec in specs:
        if spec.name in registry:
            raise ValueError(f"feature {spec.name} is defined twice")
        registry[spec.name] = spec
    return registry


# --- the table ---

# the 24 production ships the regression DNN inputs without the per-jet tagger scores

NO_TAGGER_SCORES = frozenset({"24"})

FOUR_VECTOR = ("px", "py", "pz", "e")
TAGGER_SCORES = ("tag_b", "tag_cvsb", "tag_cvsl")
PNET_ERAS = frozenset({"22pre", "22post", "23pre", "23post"})


FEATURES: dict[str, FeatureSpec] = _registry(
    # --- continuous inputs ---
    # missing transverse energy, always present
    *components("met", names=("px", "py")),
    *components("met", names=("cov00", "cov01", "cov11")),
    # visible tau decay products, always present
    *components("vis_tau1", names=FOUR_VECTOR),
    *components("vis_tau2", names=FOUR_VECTOR),
    # b-jet candidates: four-vector 0 and scores -1 when no jet pair is selected
    *components("bjet1", names=FOUR_VECTOR, dummy=0.0),
    *components("bjet1", names=TAGGER_SCORES, dummy=-1.0, unavailable_in=NO_TAGGER_SCORES),
    *components("bjet1", names=("hhbtag",), dummy=-1.0),
    *components("bjet2", names=FOUR_VECTOR, dummy=0.0),
    *components("bjet2", names=TAGGER_SCORES, dummy=-1.0, unavailable_in=NO_TAGGER_SCORES),
    *components("bjet2", names=("hhbtag",), dummy=-1.0),
    # fat jet, 0 when no fat jet is selected
    *components("fatjet", names=FOUR_VECTOR, dummy=0.0),
    # composite systems; everything built from the b-jets or the fat jet inherits their marker
    *components("htt", names=FOUR_VECTOR),
    *components("hbb", names=FOUR_VECTOR, dummy=0.0),
    *components("htthbb", names=FOUR_VECTOR, dummy=0.0),
    *components("httfatjet", names=FOUR_VECTOR, dummy=0.0),
    # regressed neutrinos, always present, no energy component
    *components("nu1", names=("px", "py", "pz")),
    *components("nu2", names=("px", "py", "pz")),

    # --- categorical inputs ---
    FeatureSpec(name="pair_type", kind="categorical", values=(0, 1, 2), column=None),
    # decay modes: 0, 1 = 1-prong (-1 = e/mu in dm1), 10, 11 = 3-prong; the groups sort the embedding monitoring
    FeatureSpec(name="dm1", kind="categorical", values=(-1, 0, 1, 10, 11), column=None,
                value_groups={"1prong": (-1, 0, 1), "3prong": (10, 11)}),
    FeatureSpec(name="dm2", kind="categorical", values=(0, 1, 10, 11), column=None,
                value_groups={"1prong": (0, 1), "3prong": (10, 11)}),
    FeatureSpec(name="vis_tau1_charge", kind="categorical", values=(-1, 1), column=None),
    FeatureSpec(name="vis_tau2_charge", kind="categorical", values=(-1, 1), column=None),
    FeatureSpec(name="has_jet_pair", kind="categorical", values=(0, 1), column=None),  # whether two or more jets are present
    FeatureSpec(name="has_fatjet", kind="categorical", values=(0, 1), column=None),  # whether a selected fatjet is present
    FeatureSpec(name="channel_id", kind="categorical", values=(1, 2, 3), column="channel_id"),  # bare column, not prefixed
    FeatureSpec(name="bjet1_upart_b", kind="continuous", dummy=-1.0, jagged=(ERAS["24"].btag_score_branch, 0), unavailable_in=PNET_ERAS),
    FeatureSpec(name="bjet2_upart_b", kind="continuous", dummy=-1.0, jagged=(ERAS["24"].btag_score_branch, 1), unavailable_in=PNET_ERAS),

    # --- derived inputs, calculated by the reader, see data_handling/derived_features.py ---
    *(FeatureSpec(name=name, kind="continuous", derived=True) for name in ALWAYS_PRESENT),
    # these need the b-jet candidates and carry the marker when a candidate is missing
    *(FeatureSpec(name=name, kind="continuous", derived=True, dummy=MISSING) for name in NEEDS_BJETS),
    # these need the collection of all jets, which the production does not ship yet
    *(FeatureSpec(name=name, kind="continuous", derived=True) for name in JET_FEATURES),
    # stand-in for n_btags from the two HH b-jet candidates only
    *(FeatureSpec(name=name, kind="continuous", derived=True) for name in HHBJET_FEATURES),
)
