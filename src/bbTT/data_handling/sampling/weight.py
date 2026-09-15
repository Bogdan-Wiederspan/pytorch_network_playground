from dataclasses import dataclass

import torch


@dataclass
class WeightStatistics:
    """
    Typed replacement for the old nested
    {"normalization_weights": {"whole_sum": ...}, "product_of_weights": {...}} dict.
    Produced by WeightAggregator, consumed by Process / ProcessSampler.
    """
    normalization_whole_sum: torch.Tensor
    normalization_training_sum: torch.Tensor
    normalization_validation_sum: torch.Tensor
    normalization_test_sum: torch.Tensor
    normalization_evaluation_sum: torch.Tensor

    product_whole_sum: torch.Tensor
    product_training_sum: torch.Tensor
    product_validation_sum: torch.Tensor
    product_test_sum: torch.Tensor
    product_evaluation_sum: torch.Tensor

    @property
    def per_event_norm(self) -> float:
        """Total normalization weight of the whole process (scalar)."""
        return self.normalization_whole_sum.item()

class WeightAggregator:
    """
    Accumulates and aggregates process weights from events.
    Process type is always read from uid[0] — never re-derived from process_id.
    """

    _PHASES = ("training", "validation", "test")

    def __init__(self, events: dict, indices: dict):
        self.weights: dict[tuple[str, str], WeightStatistics] = self._calculate_process_weights(events, indices)

    def _calculate_process_weights(self, events, indices) -> dict[tuple[str, str], WeightStatistics]:
        weights = {}
        for uid, _events in events.items():
            norm_w = _events["normalization_weights"]
            prod_w = _events["product_of_weights"]
            mask = _events["evaluation_mask"]
            idx = indices[uid]

            weights[uid] = WeightStatistics(
                normalization_whole_sum=norm_w.sum(),
                normalization_training_sum=norm_w[idx["training"]].sum(),
                normalization_validation_sum=norm_w[idx["validation"]].sum(),
                normalization_test_sum=norm_w[idx["test"]].sum(),
                normalization_evaluation_sum=norm_w[mask].sum(),
                product_whole_sum=prod_w.sum(),
                product_training_sum=prod_w[idx["training"]].sum(),
                product_validation_sum=prod_w[idx["validation"]].sum(),
                product_test_sum=prod_w[idx["test"]].sum(),
                product_evaluation_sum=prod_w[mask].sum(),
            )
        return weights

    def sum_by_process_type(self, weight_kind: str, phase: str = "whole") -> dict[str, torch.Tensor]:
        """
        Replaces process_weights_sum_from_nested_weight + calculate_summary_statistics_from_process_weights.
        weight_kind: "normalization" or "product"
        phase: "whole", "training", "validation", "test", or "evaluation"
        Groups by uid[0] (the real process_type) instead of guessing from pid.
        """
        field_name = f"{weight_kind}_{phase}_sum"
        totals: dict[str, torch.Tensor] = {}
        for uid, stats in self.weights.items():
            process_type = uid[0]
            value = getattr(stats, field_name)
            totals[process_type] = totals.get(process_type, torch.tensor(0.0)) + value
        return totals

    def __call__(self, weight_kind: str, phase: str = "whole") -> dict[str, torch.Tensor]:
        return self.sum_by_process_type(weight_kind, phase)
