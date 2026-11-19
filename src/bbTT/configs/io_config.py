from __future__ import annotations

import hashlib
import os
import pathlib
from dataclasses import dataclass, field
from string import Formatter
from typing import Literal, Optional, get_args

from bbTT.data_handling.eras import ERAS, column_prefix, era_order
from bbTT.data_handling.features import FEATURES
from bbTT.data_handling.io.find_datasets import cached_find_datasets

# eras that can be configured chosen from, in chronological order
# the related data of an era is stored in data_handling/eras.py
ERAS_CHOICE = Literal["22pre", "22post", "23pre", "23post", "24"]



@dataclass
class DataConfig:
    target_map: dict[str, int] = field(default_factory=lambda: {"hh": 0, "tt": 1, "dy": 2})  # node: index

    # features for the standardization layer path, one list per era that needs its own ("default" for all
    # others); which list applies follows from ``eras``, see ``continuous_features``
    continuous_features_by_era: dict[str, tuple[str, ...]] = field(default_factory=lambda: {
        "default": (
            "met_px", "met_py",
            "met_cov00", "met_cov01", "met_cov11",
            "vis_tau1_px", "vis_tau1_py", "vis_tau1_pz", "vis_tau1_e",
            "vis_tau2_px", "vis_tau2_py", "vis_tau2_pz", "vis_tau2_e",
            "bjet1_px", "bjet1_py", "bjet1_pz", "bjet1_e", "bjet1_tag_b", "bjet1_tag_cvsb", "bjet1_tag_cvsl", "bjet1_hhbtag",
            "bjet2_px", "bjet2_py", "bjet2_pz", "bjet2_e", "bjet2_tag_b", "bjet2_tag_cvsb", "bjet2_tag_cvsl", "bjet2_hhbtag",
            "fatjet_px", "fatjet_py", "fatjet_pz", "fatjet_e",
            "htt_e", "htt_px", "htt_py", "htt_pz",
            "hbb_e", "hbb_px", "hbb_py", "hbb_pz",
            "htthbb_e", "htthbb_px", "htthbb_py", "htthbb_pz",
            "httfatjet_e", "httfatjet_px", "httfatjet_py", "httfatjet_pz",
            "nu1_px", "nu1_py", "nu1_pz",
            "nu2_px", "nu2_py", "nu2_pz",
        ),
        # 24: the ParticleNet tagger scores are replaced by the UParT b score split from the jagged candidate branch
        "24": (
            "met_px", "met_py",
            "met_cov00", "met_cov01", "met_cov11",
            "vis_tau1_px", "vis_tau1_py", "vis_tau1_pz", "vis_tau1_e",
            "vis_tau2_px", "vis_tau2_py", "vis_tau2_pz", "vis_tau2_e",
            "bjet1_px", "bjet1_py", "bjet1_pz", "bjet1_e", "bjet1_upart_b", "bjet1_hhbtag",
            "bjet2_px", "bjet2_py", "bjet2_pz", "bjet2_e", "bjet2_upart_b", "bjet2_hhbtag",
            "fatjet_px", "fatjet_py", "fatjet_pz", "fatjet_e",
            "htt_e", "htt_px", "htt_py", "htt_pz",
            "hbb_e", "hbb_px", "hbb_py", "hbb_pz",
            "htthbb_e", "htthbb_px", "htthbb_py", "htthbb_pz",
            "httfatjet_e", "httfatjet_px", "httfatjet_py", "httfatjet_pz",
            "nu1_px", "nu1_py", "nu1_pz",
            "nu2_px", "nu2_py", "nu2_pz",
        ),
    })

    categorical_features: tuple[str] = (
        "pair_type",
        # "_channel_id",
        "dm1",
        "dm2",
        "vis_tau1_charge",
        "vis_tau2_charge",
        "has_jet_pair",
        "has_fatjet",
    )  # features for the embedding path

    dataset_pattern: tuple[str] = (
        "dy_*",
        "tt_*",
        "hh_ggf_hbb_htt_kl1_kt1*",
    )  # pattern to find datasets

    # eras: tuple[ERAS_CHOICE] = ("22pre", "22post", "23pre", "23post",)  # ERAS that are trained on. For each era a separate cache is created
    eras: tuple[ERAS_CHOICE] = ("24",)  # ERAS that are trained on. For each era a separate cache is created

    flush_threshold: int = 1_000_000  # number of rows before a flush is initiated when creating the cache, reduce when memory issues are a problem
    statistics_chunk_rows: int = 1_000_000  # rows per chunk when computing feature mean/std; bounds the transient memory, does not change the result

    cuts: Optional[tuple[str]] = (
        # "({tau2_isolated} == 1)", # exist only for <prod28
        # "(({channel_id} == 1) | ({channel_id} == 2) | ({channel_id} == 3) )", # < prod 28
        "({leptons_os} == 1)",
        "((({channel_id} == 1) & ({num_taus_iso} >= 1) | ({channel_id} == 2) &  ({num_taus_iso} >= 1)| ({channel_id} == 3) &  ({num_taus_iso} >= 1)))",  # noqa
        "(({vis_tau1_charge} == 1) | ({vis_tau1_charge} == -1))",
        "(({vis_tau2_charge} == 1) | ({vis_tau2_charge} == -1))",
        "({vbf_dnn_moe_hh_vbf} < 0.5)",
    )  # templates, {name} is resolved to the array column by uproot_cuts


    # --- Helpers ---

    @property
    def data_prefix(self) -> str:
        """
        Prefix of the feature columns for the current production, e.g. "res_dnn_pnet" or "reg_dnn_moe".
        Derived on access from the production in INPUT_DATA_DIR (see data_handling/eras.py).

        Returns:
            str: Column prefix.
        """
        return column_prefix(pathlib.Path(os.environ["INPUT_DATA_DIR"]).stem)

    @property
    def datasets(self) -> tuple[str, ...]:
        """
        All dataset file paths matching ``dataset_pattern`` for the configured ``eras``.
        Derived on access through the memoized ``cached_find_datasets``, so it always follows the current
        ``eras`` (e.g. after a resume override) and is never stored in a checkpoint.

        Returns:
            tuple[str, ...]: Dataset file paths.
        """
        return cached_find_datasets(tuple(self.dataset_pattern), tuple(self.eras), file_type="root")


    @property
    def continuous_features(self) -> tuple[str, ...]:
        """
        The continuous features of the configured eras: the entry of ``continuous_features_by_era``
        for the era, "default" for eras without their own. All configured eras must resolve to the same list.

        Returns:
            tuple[str, ...]: Feature names in tensor column order.

        Raises:
            ValueError: If the configured eras resolve to different feature lists.
        """
        by_era = self.continuous_features_by_era
        keys = {era if era in by_era else "default" for era in self.eras}
        if len(keys) != 1:
            raise ValueError(f"eras {self.eras} resolve to different feature lists {sorted(keys)}")
        return tuple(by_era[keys.pop()])

    @property
    def dummy_values(self) -> dict[str, float]:
        """
        Marker of a missing value per configured continuous feature that has one, from the feature
        registry. Features that are never missing are not listed; consumers that need one value per
        tensor column use ``data_handling.utils.markers_by_position``.

        Returns:
            dict[str, float]: Feature name -> marker.
        """
        return {f: FEATURES[f].dummy for f in self.continuous_features if FEATURES[f].dummy is not None}

    @property
    def expected_embedding_inputs(self) -> dict[str, list[int]]:
        """
        Values every configured categorical feature takes in the data, from the feature registry.

        Returns:
            dict[str, list[int]]: Feature name -> values, consumed by the tokenizer.
        """
        return {f: list(FEATURES[f].values) for f in self.categorical_features}

    def prefixed(self, name: str) -> str:
        """
        Actual array column of a name. Registered features resolve through their spec (production
        prefix or explicit column); any other name, e.g. a cut field such as "leptons_os", is returned unchanged.

        Args:
            name (str): Feature name or bare column name.

        Returns:
            str: Column name, e.g. "reg_dnn_moe_bjet1_px".
        """
        spec = FEATURES.get(name)
        return spec.column_name(self.data_prefix) if spec is not None else name

    @property
    def uproot_continuous_columns(self) -> list[str]:
        """
        Array columns of ``continuous_features``, in config order.

        Returns:
            list[str]: Column names.
        """
        return [self.prefixed(f) for f in self.continuous_features]

    @property
    def uproot_categorical_columns(self) -> list[str]:
        """
        Array columns of ``categorical_features``, in config order.

        Returns:
            list[str]: Column names.
        """
        return [self.prefixed(f) for f in self.categorical_features]

    def resolve_template(self, template: str) -> str:
        """
        Replace every {name} placeholder of a cut template by its array column.

        Args:
            template (str): Cut with placeholders, e.g. "({vis_tau1_charge} == 1)".

        Returns:
            str: Cut with column names.
        """
        formatter = Formatter()
        for _, field_name, _, _ in formatter.parse(template):
            if field_name is not None:
                template = template.replace(f"{{{field_name}}}", self.prefixed(field_name))
        return template

    @property
    def uproot_cuts(self) -> list[str]:
        """
        The cuts with placeholders resolved to array columns.

        Returns:
            list[str]: Resolved cuts.
        """
        return [self.resolve_template(cut) for cut in self.cuts]

    @property
    def production(self):
        """
        Return the production name, e.g. "prod14" or "prod19", derived from the input data directory.
        """
        return os.environ["ERA"]

    def cache_entries(self):
        """
        Return a dictionary of the config that defines uniquely the dataset.
        This is relevant for caching algorithms, as changes in this config will create a new hash of the data.
        """
        return {
            "production": self.production,
            "target_map": self.target_map,
            "continuous_features": self.continuous_features,
            "categorical_features": self.categorical_features,
            "expected_embedding_inputs": self.expected_embedding_inputs,
            "datasets": self.dataset_pattern,
            "cuts": self.cuts,
        }

    def content_hash(self):
        """
        Return a hash for each era the config has.
        This is relevant for caching algorithms, as changes in this config will create a new hash of the data.
        """
        hashable_dict = sorted(self.cache_entries().items(), key=lambda item: item[0])
        h = tuple(hashable_dict)
        return hashlib.sha256(str(h).encode("utf-8")).hexdigest()[:10]

    @property
    def ordered_eras(self) -> tuple[str, ...]:
        """
        The configured eras in chronological order (order of ERAS_CHOICE), independent of
        how ``eras`` is written. The era caches are merged in this order, which fixes the
        event order inside each uid and therefore the k-fold split.

        Returns:
            tuple[str, ...]: Sorted copy of ``eras``.
        """
        return tuple(sorted(self.eras, key=era_order))

    @property
    def uproot_jagged_columns(self) -> dict[str, tuple[str, int, float]]:
        """
        Configured features that are one entry of a jagged branch, for the root reader to split.

        Returns:
            dict[str, tuple[str, int, float]]: Flat column name -> (branch, index, fill value for a missing entry).
        """
        return {
            self.prefixed(f): (*FEATURES[f].jagged, FEATURES[f].dummy)
            for f in self.continuous_features + self.categorical_features
            if FEATURES[f].jagged is not None
        }

    def __post_init__(self):
        if get_args(ERAS_CHOICE) != tuple(ERAS):
            raise RuntimeError(f"ERAS_CHOICE {get_args(ERAS_CHOICE)} does not match the eras known in data_handling/eras.py {tuple(ERAS)}")

        # eras must be known and share one b-tagger: tag scores of different taggers are not comparable,
        # so e.g. 24 (UParT) can never be cached and trained together with 22/23 (ParticleNet)
        unknown_eras = set(self.eras) - set(ERAS)
        if unknown_eras:
            raise ValueError(f"Unknown eras {sorted(unknown_eras)}, known: {list(ERAS)}")
        btaggers = {ERAS[era].btagger for era in self.eras}
        if len(btaggers) > 1:
            raise ValueError(
                f"eras {self.eras} mix b-taggers {sorted(btaggers)}; "
                "only eras with the same b-tagger can be trained together"
            )

        # every configured feature must be registered with the right kind and exist in every configured era;
        # the lists are never trimmed here, continuous_features_by_era holds one per era that differs
        for kind, names in (("continuous", self.continuous_features), ("categorical", self.categorical_features)):
            unknown = [f for f in names if f not in FEATURES]
            if unknown:
                raise ValueError(f"{kind}_features not in the feature registry: {unknown}")
            wrong_kind = [f for f in names if FEATURES[f].kind != kind]
            if wrong_kind:
                raise ValueError(f"{kind}_features contains features of another kind: {wrong_kind}")
            for era in self.eras:
                missing = [f for f in names if not FEATURES[f].is_available(era)]
                if missing:
                    raise ValueError(f"{kind}_features {missing} are not available in era {era}")
