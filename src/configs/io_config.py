from __future__ import annotations

import functools
import os
import pathlib
from dataclasses import dataclass, field
from string import Formatter
from typing import Any, Dict, List, Literal, Optional, Tuple

from data_handling.utils import find_datasets

ERAS_CHOICE = Literal["22pre", "22post", "23pre", "23post"]


@dataclass
class DataConfig:
    # changes in this config will create a NEW hash of the data
    target_map: Dict[str, int] = field(default_factory=lambda: {
        "hh": 0, "tt": 1, "dy": 2
        }) # node: index

    continuous_features: Tuple[str] = (
        "met_px", "met_py",
        "met_cov00", "met_cov01", "met_cov11",
        "vis_tau1_px", "vis_tau1_py", "vis_tau1_pz", "vis_tau1_e",
        "vis_tau2_px", "vis_tau2_py", "vis_tau2_pz", "vis_tau2_e",
        "bjet1_px", "bjet1_py", "bjet1_pz", "bjet1_e",
        "bjet1_tag_b", "bjet1_tag_cvsb", "bjet1_tag_cvsl", "bjet1_hhbtag",
        "bjet2_px", "bjet2_py", "bjet2_pz", "bjet2_e",
        "bjet2_tag_b", "bjet2_tag_cvsb", "bjet2_tag_cvsl", "bjet2_hhbtag",
        "fatjet_px", "fatjet_py", "fatjet_pz", "fatjet_e",
        "htt_e", "htt_px", "htt_py", "htt_pz",
        "hbb_e", "hbb_px", "hbb_py", "hbb_pz",
        "htthbb_e", "htthbb_px", "htthbb_py", "htthbb_pz",
        "httfatjet_e", "httfatjet_px", "httfatjet_py", "httfatjet_pz",
        "nu1_px", "nu1_py", "nu1_pz",
        "nu2_px", "nu2_py", "nu2_pz",
    )

    categorical_features: Tuple[str] = (
        "pair_type",
        # "_channel_id",
        "dm1",
        "dm2",
        "vis_tau1_charge",
        "vis_tau2_charge",
        "has_jet_pair",
        "has_fatjet",
    )

    expected_embedding_inputs: Dict[str, List[Any]] = field(default_factory=lambda: {
        "pair_type": [0, 1, 2],  # see mapping below
        "dm1": [-1, 0, 1, 10, 11],  # -1 for e/mu
        "dm2": [0, 1, 10, 11],
        "vis_tau1_charge": [-1, 1],
        "vis_tau2_charge": [-1, 1],
        "has_fatjet": [0, 1],  # whether a selected fatjet is present
        "has_jet_pair": [0, 1],  # whether two or more jets are present
        "year_flag": [0, 1, 2, 3, 4, 5, 6, 7], # 0: 2016APV, 1: 2016, 2: 2017, 3: 2018, 4: 2022preEE, 5: 2022postEE, 6: 2023pre, 7: 2023post #noqa
        "channel_id": [1, 2, 3],
    })

    dataset_pattern: Tuple[str] = (
        # "dy_*",
        "dy_m50toinf_0j_amcatnlo",
        "tt_fh_powheg",
        # "tt_*",
        "hh_ggf_hbb_htt_kl1_kt1*",
        # "hh_ggf_hbb_htt_kl0_kt1*",
        )

    eras: Tuple[ERAS_CHOICE] = ("22pre",)
    datasets: Optional[List[str]] = field(init=False)
    cuts: Optional[Tuple[str]] = (
        "({tau2_isolated} == 1)",
        "({leptons_os} == 1)",
        "(({channel_id} == 1) | ({channel_id} == 2) | ({channel_id} == 3))",
        "({vis_tau1_charge} == 1) | ({vis_tau1_charge} == -1)",
        "({vis_tau2_charge} == 1) | ({vis_tau2_charge} == -1)",
    )

    # derived in __post_init__: cuts with placeholders resolved to actual array column names

    dummy_values: int = -99999 # value used to fill in missing values
    data_prefix: Optional[str] = field(init=False) # prefix for features, e.g. "res_dnn_pnet" or "reg_dnn_moe"

    # --- Helpers ---
    def _prefix_map(self):
        stem = pathlib.Path(os.environ["INPUT_DATA_DIR"]).stem
        stem_to_prefix = {
        "prod14": "res_dnn_pnet",
        "prod20_vbf": "reg_dnn_moe",
        "prod20": "reg_dnn_moe",
        "prod19": "res_dnn_pnet",
        "prod24" : "reg_dnn_moe",
        "prod27_dyext" : "reg_dnn_moe",
        }
        try:
            return stem_to_prefix[stem]
        except KeyError:
            raise KeyError(
                f"No feature prefix registered for INPUT_DATA_DIR stem: {stem}."
                f" known stems: {sorted(stem_to_prefix)}"
                )

    def prefixed(self, string):
        """
        Return the actual array column name for a feature, e.g. "res_dnn_pnet_bjet1_px" for "bjet1_px"
        """
        return self.column_map.get(string, string)

    @functools.cached_property
    def column_map(self) -> Dict[str, str]:
        """plain name -> actual array column name, for every feature this config knows about."""
        return {
            # scores of normal network
            "score_dy": "run3_dnn_moe_dy",
            "score_hh": "run3_dnn_moe_hh",
            "score_tt": "run3_dnn_moe_tt",
            # inputs used of normale network
            "bjet1_e" : f"{self.data_prefix}_bjet1_e",
            "bjet1_hhbtag" : f"{self.data_prefix}_bjet1_hhbtag",
            "bjet1_px" : f"{self.data_prefix}_bjet1_px",
            "bjet1_py" : f"{self.data_prefix}_bjet1_py",
            "bjet1_pz" : f"{self.data_prefix}_bjet1_pz",
            "bjet1_tag_b" : f"{self.data_prefix}_bjet1_tag_b",
            "bjet1_tag_cvsb" : f"{self.data_prefix}_bjet1_tag_cvsb",
            "bjet1_tag_cvsl" : f"{self.data_prefix}_bjet1_tag_cvsl",
            "bjet2_e" : f"{self.data_prefix}_bjet2_e",
            "bjet2_hhbtag" : f"{self.data_prefix}_bjet2_hhbtag",
            "bjet2_px" : f"{self.data_prefix}_bjet2_px",
            "bjet2_py" : f"{self.data_prefix}_bjet2_py",
            "bjet2_pz" : f"{self.data_prefix}_bjet2_pz",
            "bjet2_tag_b" : f"{self.data_prefix}_bjet2_tag_b",
            "bjet2_tag_cvsb" : f"{self.data_prefix}_bjet2_tag_cvsb",
            "bjet2_tag_cvsl" : f"{self.data_prefix}_bjet2_tag_cvsl",
            "dm1" : f"{self.data_prefix}_dm1",
            "dm2" : f"{self.data_prefix}_dm2",
            "fatjet_e" : f"{self.data_prefix}_fatjet_e",
            "fatjet_px" : f"{self.data_prefix}_fatjet_px",
            "fatjet_py" : f"{self.data_prefix}_fatjet_py",
            "fatjet_pz" : f"{self.data_prefix}_fatjet_pz",
            "has_fatjet" : f"{self.data_prefix}_has_fatjet",
            "has_jet_pair" : f"{self.data_prefix}_has_jet_pair",
            "hbb_e" : f"{self.data_prefix}_hbb_e",
            "hbb_px" : f"{self.data_prefix}_hbb_px",
            "hbb_py" : f"{self.data_prefix}_hbb_py",
            "hbb_pz" : f"{self.data_prefix}_hbb_pz",
            "htt_e" : f"{self.data_prefix}_htt_e",
            "htt_px" : f"{self.data_prefix}_htt_px",
            "htt_py" : f"{self.data_prefix}_htt_py",
            "htt_pz" : f"{self.data_prefix}_htt_pz",
            "httfatjet_e" : f"{self.data_prefix}_httfatjet_e",
            "httfatjet_px" : f"{self.data_prefix}_httfatjet_px",
            "httfatjet_py" : f"{self.data_prefix}_httfatjet_py",
            "httfatjet_pz" : f"{self.data_prefix}_httfatjet_pz",
            "htthbb_e" : f"{self.data_prefix}_htthbb_e",
            "htthbb_px" : f"{self.data_prefix}_htthbb_px",
            "htthbb_py" : f"{self.data_prefix}_htthbb_py",
            "htthbb_pz" : f"{self.data_prefix}_htthbb_pz",
            "met_cov00" : f"{self.data_prefix}_met_cov00",
            "met_cov01" : f"{self.data_prefix}_met_cov01",
            "met_cov11" : f"{self.data_prefix}_met_cov11",
            "met_px" : f"{self.data_prefix}_met_px",
            "met_py" : f"{self.data_prefix}_met_py",
            "nu1_px" : f"{self.data_prefix}_nu1_px",
            "nu1_py" : f"{self.data_prefix}_nu1_py",
            "nu1_pz" : f"{self.data_prefix}_nu1_pz",
            "nu2_px" : f"{self.data_prefix}_nu2_px",
            "nu2_py" : f"{self.data_prefix}_nu2_py",
            "nu2_pz" : f"{self.data_prefix}_nu2_pz",
            "pair_type" : f"{self.data_prefix}_pair_type",
            "vis_tau1_charge" : f"{self.data_prefix}_vis_tau1_charge",
            "vis_tau1_e" : f"{self.data_prefix}_vis_tau1_e",
            "vis_tau1_px" : f"{self.data_prefix}_vis_tau1_px",
            "vis_tau1_py" : f"{self.data_prefix}_vis_tau1_py",
            "vis_tau1_pz" : f"{self.data_prefix}_vis_tau1_pz",
            "vis_tau2_charge" : f"{self.data_prefix}_vis_tau2_charge",
            "vis_tau2_e" : f"{self.data_prefix}_vis_tau2_e",
            "vis_tau2_px" : f"{self.data_prefix}_vis_tau2_px",
            "vis_tau2_py" : f"{self.data_prefix}_vis_tau2_py",
            "vis_tau2_pz" : f"{self.data_prefix}_vis_tau2_pz",
        }

    @property
    def uproot_continuous_columns(self) -> List[str]:
        return [self.column_map[f] for f in self.continuous_features]

    @property
    def uproot_categorical_columns(self) -> List[str]:
        return [self.column_map[f] for f in self.categorical_features]

    def resolve_template(self, template: str) -> str:
        formatter = Formatter()
        for _, field_name, _, _ in formatter.parse(template):
            if field_name is not None:
                resolved_name = self.prefixed(field_name)
                template = template.replace(f"{{{field_name}}}", resolved_name)
        return template

    @property
    def uproot_cuts(self):
        """
        Return the cuts with placeholders resolved to actual array column names.
        """
        return [self.resolve_template(cut) for cut in self.cuts]

    def cache_entries(self):
        """
        Return a dictionary of the config that defines uniquely the dataset.
        This is relevant for caching algorithms, as changes in this config will create a new hash of the data.
        """
        return {
            "target_map": self.target_map,
            "continuous_features": self.continuous_features,
            "categorical_features": self.categorical_features,
            "expected_embedding_inputs": self.expected_embedding_inputs,
            "datasets": self.dataset_pattern,
            "cuts": self.cuts,
        }

    def __hash__(self):
        """
        Return a hash for each era the config has.
        This is relevant for caching algorithms, as changes in this config will create a new hash of the data.
        """
        eras = self.eras
        unique_entries = self.cache_entries()
        return hash(frozenset(self.cache_entries().items()))

    def __post_init__(self):
        # a dictionary of all files corresponding to a certain dataset
        self.data_prefix = self._prefix_map()
        self.datasets = find_datasets(self.dataset_pattern, self.eras, file_type="root", verbose=False)
