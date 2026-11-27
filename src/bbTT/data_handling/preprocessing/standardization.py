from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import TYPE_CHECKING, Any, Iterator

import torch

from bbTT.data_handling.cache import BaseCacher
from bbTT.data_handling.utils import markers_by_position
from bbTT.monitoring.logger.logger import get_logger

logger_inst = get_logger(__name__)


if TYPE_CHECKING:
    from bbTT.configs.io_config import DataConfig
    from bbTT.data_handling.sampling.sampler import ProcessSampler


class FeatureStatisticCache(BaseCacher):
    def __init__(
        self,
        dataset_config: DataConfig,
        sampler: ProcessSampler,
        split_fingerprint: dict[str, Any],
        statistic: str = "mean_std",
        statistic_config: Any = None,
        return_dummy: bool = False,
        verbose: bool = True
    ):
        """
        Provides the per-feature normalization statistics, cached on disk.

        On construction the statistics are either loaded from the cache file belonging
        to the config hash and fingerprint, or computed from the sampler and written there.
        The cache path is ``$CACHE_DIR/<content_hash>/training_statistics_<fingerprint_has>.json``.
        The fingerprint covers everything that changes the numbers but is not part the dataset definitions (era, split, etc.).
        This ensures that stale statistics are not silently reused when the dataset is changed, and that different splits of the same dataset do not

        The two numbers per feature are always called ``mean`` and ``std``, since this is what the
        standardization layer subtracts and divides by. What they are depends on *statistic*:
        "mean_std" gives the weighted mean and standard deviation, "quantile" the median and the
        quantile width, see ``compute_quantile_statistics``.

        Args:
            dataset_config (DataConfig): Config providing ``content_hash()``, ``continuous_features`` and ``dummy_values``.
            sampler (ProcessSampler): Sampler holding the registered processes.
            split_fingerprint (dict[str, Any]): Split parameters, see ``FoldAndSplitCoordinator.fingerprint``.
            statistic (str): "mean_std" or "quantile", see ``StandardizationConfig.statistic``.
            statistic_config (Any): Settings of the chosen statistic, see ``StandardizationConfig.active_config``.
                A dataclass instance, e.g. QuantileStatisticConfig. None is only valid for "mean_std", which has no settings.
            return_dummy (bool): Skip the computation and use an identity transform (mean 0, std 1). Useful when pretrained weights are loaded anyway.
            verbose (bool): Log the resulting per-feature statistics.

        Raises:
            FileExistsError: If the cache root directory does not exist.
        """
        super().__init__(config=dataset_config)
        self.sampler = sampler
        self.split_fingerprint = split_fingerprint
        self.statistic = statistic
        self.statistic_config = statistic_config
        self.return_dummy = return_dummy

        self.features = self.config.continuous_features
        self.padding_dummy = markers_by_position(self.features, self.config.dummy_values)

        self.fingerprint = self.build_fingerprint(split_fingerprint)
        self.path = self.cache_root / self.hash / f"training_statistics_{self.fingerprint_hash()}.json"

        self.mean, self.std = None, None
        if not self.path.exists():
            self.mean, self.std = self.get_statistics()
            self.save()
        else:
            self.mean, self.std = self.load()

        if verbose:
            self.verbose_statistics()

    def load(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reads mean and std from the cache file.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Mean and std as float32 tensors.
        """
        content = json.loads(self.path.read_text())
        mean = torch.tensor(content["mean"], dtype=torch.float32)
        std = torch.tensor(content["std"], dtype=torch.float32)
        logger_inst.info(f"Loaded mean and std from file {self.path}")
        return mean, std

    def save(self):
        """
        Writes the current statistics to the cache file.

        The features and padding values are stored alongside the numbers so the
        file stays self-describing and can be inspected without the config.
        """

        content = {
            "mean": self.mean.numpy().tolist(),
            "std": self.std.numpy().tolist(),
            "padding": self.padding_dummy,
            "features": self.features,
            "fingerprint": self.fingerprint,
        }

        tmp_path = self.path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(content, indent=2))
        tmp_path.replace(self.path)
        logger_inst.info(f"Saved normalization statistics to {self.path}")

    def build_fingerprint(self, split_fingerprint: dict[str, Any]) -> dict[str, Any]:
        """
        Collects everything that determines the numbers of the statistics.

        The relative weights are stored as result instead of listing their inputs (batch size,
        sample ratios, luminosity, ...), so any upstream change that shifts a weight is
        detected automatically.

        Args:
            split_fingerprint (dict[str, Any]): Split parameters, see ``FoldAndSplitCoordinator.fingerprint``.

        Returns:
            dict[str, Any]: JSON-serializable fingerprint.
        """
        processes = self.sampler.registry.all()  # get all registered processes, dict[uid, Process]
        return {
            "eras": list(self.config.eras),
            "features": list(self.features),
            "padding": self.padding_dummy,
            "return_dummy": self.return_dummy,
            # only the chosen statistic and its own settings: the settings of the other statistics
            # are not part of the fingerprint, so changing them never invalidates this cache
            "active_statistic": self.statistic,
            "active_config": dataclasses.asdict(self.statistic_config) if self.statistic_config is not None else {},
            "split": split_fingerprint,
            # rounded to 8 so that the last float bits must not decide between cache hit and miss
            "relative_weights": {
                "/".join(map(str, uid)): float(f"{proc.relative_weight:.8g}") for uid, proc in processes.items()
            },
        }

    def fingerprint_hash(self) -> str:
        """
        Short stable hash of the fingerprint, used in the file name.

        Uses sha256 (like ``DataConfig.content_hash``) and not the builtin ``hash``, which is
        randomized per process for strings. ``sort_keys`` makes the result independent of the
        dict order. Only the first 10 hex characters (40 bit) are kept (approx collision between 2 files 1e-11).

        Returns:
            str: 10 hex characters.
        """
        payload = json.dumps(self.fingerprint, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]

    def get_statistics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Dispatches to the dummy or the real computation of the configured statistic.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Mean and std per feature (center and scale for "quantile").

        Raises:
            ValueError: If the configured statistic is unknown.
        """
        if self.return_dummy:
            mean, std = self.dummy_statistics()
        elif self.statistic == "mean_std":
            mean, std = self.compute_statistics()
        elif self.statistic == "quantile":
            mean, std = self.compute_quantile_statistics()
        else:
            raise ValueError(f"Unknown standardization statistic {self.statistic!r}, expected 'mean_std' or 'quantile'")
        return mean, std

    def dummy_statistics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Builds statistics describing an identity transformation.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Zeros and ones, shape [num_features].
        """
        num_f = len(self.features)
        mean = torch.zeros(num_f)
        std = torch.ones(num_f)
        logger_inst.info(
            "\nNo normalization statistics are calculated, since return_dummy is True."
            "Returning dummy values: mean = 0 and std = 1"
        )
        return mean, std

    def __str__(self) -> str:
        return self._str()

    def verbose_statistics(self):
        """
        Logs one line per feature with its mean and width at debug level.
        """

        msg = [f"statistic: {self.statistic} (for 'quantile': mean = median, std = quantile width)"]
        for f_name, f_mean, f_std in zip(self.features, self.mean, self.std):
            msg.append(f"{f_name:<30}: mean:{f_mean:>10.4} std:{f_std:>10.4} var:{f_std**2:>10.4}")
        msg = "\n" + "\n".join(msg)
        logger_inst.debug(msg)

    def compute_statistics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the weighted mean and standard deviation over all processes of the sampler.

        Each process contributes its masked mean and variance. The processes are
        combined via the law of total variance, so the result describes the weighted
        mixture the network actually sees:

            Var_total = sum_i w_i * Var_i + sum_i w_i * (mu_i - mu_total)^2
                        (within processes)   (between processes)

        Padding entries are excluded via a mask, since they are placeholders and would
        otherwise pull the mean towards the dummy value. The reduction runs in float64
        because the later square root is numerically sensitive to the large dynamic
        range of the raw four-momenta.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Weighted mean and standard deviation
                per feature, shape [num_features] each (float64).

        Raises:
            ValueError: If the number of padding values does not match the number of features,
        """
        if self.return_dummy:
            return self.dummy_statistics()
        logger_inst.info("Calculate mean and std over all subphase spaces")
        # per-process results are collected first and combined at the end,
        # since the between-term needs the total mean, which is only known afterwards
        means, variances, weights = [], [], []

        processes = self.sampler.registry.all()  # dict[uid, Process]

        for current_pid_idx, (uid, proc) in enumerate(processes.items(), start=1):
            array = proc.continuous  # [num_events, num_features]

            logger_inst.info_progress(f"\rcalculating stats for pids: {current_pid_idx}/{len(processes)}")

            num_f = array.shape[-1]

            # [num_features], broadcasts against the [rows, num_features] chunks
            ignore_tensor = torch.tensor(self.padding_dummy, dtype=array.dtype)
            if ignore_tensor.shape != torch.Size([num_f]):
                raise ValueError(f"Got {len(self.padding_dummy)} padding values for {num_f} features")


            masked_mean, masked_var, masked_count = self._masked_moments(array, ignore_tensor)
            # a feature with 1 event returns NaN for its variance -> use variance of 1
            # a feature with only padded values returns Nan for its mean --> use mean of 0 instead
            masked_mean = torch.nan_to_num(masked_mean, nan=0.0)
            masked_var = torch.nan_to_num(masked_var, nan=1.0)

            means.append(masked_mean)
            variances.append(masked_var)

            # relative weight contribution is excluding masked values, for this reason the scaling happens
            weights.append(proc.relative_weight * masked_count / array.shape[0])

        # normalized weights as column vector [num_procs, 1] to broadcast over features
        means = torch.stack(means, dim=0)          # [num_procs, num_features]
        variances = torch.stack(variances, dim=0)  # [num_procs, num_features]
        w = torch.stack(weights, dim=0)            # [num_procs, num_features]

        # a feature without a single real value over all processes cannot be standardized
        w_sum = w.sum(dim=0)
        if torch.any(w_sum == 0):
            empty = [f for f, is_empty in zip(self.features, (w_sum == 0).tolist()) if is_empty]
            raise ValueError(f"Features {empty} are fully padded over all processes of the sampler.")
        w = w / w_sum  # normalized per feature

        total_mean = (w * means).sum(dim=0)
        within = (w * variances).sum(dim=0)                       # spread inside each process
        between = (w * (means - total_mean) ** 2).sum(dim=0)      # spread of the process means
        total_std = torch.sqrt(within + between)
        return total_mean.to(torch.float32), total_std.to(torch.float32)

    def compute_quantile_statistics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates center and scale per feature from three weighted quantiles over all processes of the sampler.

        With the left and right quantile $q_l$, $q_r$ (``QuantileStatisticConfig.edges``) and the median $q_{50}$:

            center = $q_{50}$
            scale  = $((q_{50} - q_l) + (q_r - q_{50})) / 2$

        so the scale is the mean distance of the median to both sides. For a Gaussian and the edges
        (0.16, 0.84) this reproduces mean and standard deviation, for skewed features it is not driven by the tails.

        Every event of a process weighs ``relative_weight / num_events``, so a process counts as much as
        in a training batch and its events are equal among themselves. This is the composition weighting
        of the validation and the same weighting ``compute_statistics`` uses. Padding entries are
        excluded, as for mean and std. A quantile cannot be combined from per-process results, so each
        feature is collected over all processes; this runs feature by feature to bound the memory.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Center and scale per feature, shape [num_features] each (float32).

        Raises:
            ValueError: If the number of padding values does not match the number of features, if a feature
                is fully padded over all processes, or if a feature has scale 0 (both edges equal the median).
        """
        logger_inst.info("Calculate median and quantile width over all subphase spaces")
        processes = self.sampler.registry.all()  # dict[uid, Process]
        num_f = len(self.features)
        if len(self.padding_dummy) != num_f:
            raise ValueError(f"Got {len(self.padding_dummy)} padding values for {num_f} features")

        left_edge, right_edge = self.statistic_config.edges
        center = torch.empty(num_f, dtype=torch.float64)
        scale = torch.empty(num_f, dtype=torch.float64)

        for feature_idx, (feature, marker) in enumerate(zip(self.features, self.padding_dummy)):
            logger_inst.info_progress(f"\rcalculating quantiles for features: {feature_idx + 1}/{num_f}")

            values, weights = [], []
            for proc in processes.values():
                column = proc.continuous[:, feature_idx]
                real_values = column[column != marker].to(torch.float64)
                values.append(real_values)
                # the weight of an event does not change when other events of its process are padded
                event_weight = proc.relative_weight / column.shape[0]
                weights.append(torch.full_like(real_values, event_weight))
            values, weights = torch.cat(values), torch.cat(weights)

            if values.numel() == 0 or weights.sum() == 0:
                raise ValueError(f"Feature {feature} is fully padded over all processes of the sampler.")

            left, median, right = self._weighted_quantiles(values, weights, (left_edge, 0.5, right_edge))
            center[feature_idx] = median
            scale[feature_idx] = ((median - left) + (right - median)) / 2

        # a scale of 0 would divide by zero in the standardization layer
        flat = [f for f, is_flat in zip(self.features, (scale == 0).tolist()) if is_flat]
        if flat:
            raise ValueError(
                f"Features {flat} have a quantile width of 0: the quantiles {(left_edge, right_edge)} equal the median. "
                "This happens for (almost) constant or discrete features, use wider edges or 'mean_std'."
            )
        return center.to(torch.float32), scale.to(torch.float32)

    @staticmethod
    def _weighted_quantiles(
        values: torch.Tensor,
        weights: torch.Tensor,
        quantiles: tuple[float, ...],
    ) -> torch.Tensor:
        """
        Weighted quantiles of a 1D sample via the inverse of its weighted cumulative distribution.

        The values are sorted and their weights accumulated; the quantile $q$ is the smallest value
        whose cumulative weight share reaches $q$. There is no interpolation between neighbouring
        values, the result is always a value that occurs in the sample.

        Args:
            values (torch.Tensor): Sample, shape [num_values].
            weights (torch.Tensor): Non-negative weight per value, shape [num_values], need not be normalized.
            quantiles (tuple[float, ...]): Quantiles in [0, 1].

        Returns:
            torch.Tensor: Value per quantile, shape [len(quantiles)], dtype of *values*.
        """
        order = torch.argsort(values)
        sorted_values = values[order]
        cumulative = torch.cumsum(weights[order], dim=0)
        cumulative = cumulative / cumulative[-1]
        targets = torch.tensor(quantiles, dtype=cumulative.dtype)
        # float rounding can leave the last cumulative value slightly below 1
        positions = torch.searchsorted(cumulative, targets).clamp_max(sorted_values.numel() - 1)
        return sorted_values[positions]

    def _masked_moments(
        self,
        array: torch.Tensor,
        ignore_tensor: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Masked mean, unbiased variance and count of real values per feature, computed chunk-wise in float64.

        Args:
            array (torch.Tensor): Events of one process, shape [num_events, num_features].
            ignore_tensor (torch.Tensor): Marker of a padded value per feature, shape [num_features].

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Mean, variance and count, shape [num_features] each.
                Mean is NaN where count is 0, variance is NaN where count is below 2.

        """
        num_features = array.shape[-1]
        chunk_rows = self.config.statistics_chunk_rows

        def chunks() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
            # yields (float64 chunk, include mask) pairs
            # read data chunk-wise - excluding dummy values specified by ignore_tensor
            for start in range(0, array.shape[0], chunk_rows):
                chunk = array[start:start + chunk_rows]
                include_mask = chunk != ignore_tensor
                yield chunk.to(torch.float64), include_mask

        # pass 1: masked sum and count -> mean
        total = torch.zeros(num_features, dtype=torch.float64)
        count = torch.zeros(num_features, dtype=torch.float64)
        for chunk, include_mask in chunks():
            chunk.masked_fill_(~include_mask, 0.0)
            total += chunk.sum(dim=0)
            count += include_mask.sum(dim=0)
        mean = total / count

        # pass 2: masked sum of squared deviations -> unbiased variance
        squared_deviation = torch.zeros(num_features, dtype=torch.float64)
        for chunk, include_mask in chunks():
            chunk.sub_(mean).square_().masked_fill_(~include_mask, 0.0)
            squared_deviation += chunk.sum(dim=0)
        variance = squared_deviation / (count - 1)

        return mean, variance, count
