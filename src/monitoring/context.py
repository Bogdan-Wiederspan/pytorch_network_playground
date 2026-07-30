from __future__ import annotations

import fnmatch
from typing import Any

import torch


class EvalContext:
    def __init__(
        self,
        model_evaluation_state: dict[str, Any],
        predictions: torch.Tensor,
        targets: torch.Tensor,
        target_map: dict[str, int],
        event_weights: torch.Tensor,
        global_step: int,
        mode: str,
        ):
        """
        Context Object is a manager to hold everything relevant for evaluation and monitoring.
        Since this is an evaluation object no gradients are required.
        Due to plotting requirements, everything is moved to cpu.

        Args:
            model_evaluation_state (dict[str, Any]): Evaluation state of the model. Created using Model.evaluation_state.
            predictions (torch.Tensor): Prediction of the model that is used to train the network
            class_predictions (torch.Tensor): Class prediction of the model, used for monitoring
            target (torch.Tensor): Truth value to be predicted
            target_map (torch.Tensor): Mapping of the targets to their corresponding node
            event_weights (torch.Tensor): Product of alle weights, unique per event
            global_step (int): Meta Data describing the current batch step
            mode (str): Meta Data describing to which kind of data this context belongs

        """
        # core state of the model. Model it self create this state
        # model_state is detached from live model!
        self.evaluation_state = model_evaluation_state

        # core artifacts
        self.predictions = predictions.detach().cpu()
        self.targets = targets.detach().cpu()
        self.target_map = target_map
        self.event_weights = event_weights.detach().cpu()

        # meta data
        self.global_step = global_step # current batch iteration
        self.mode = mode # batch, train oder valid, influences the meta tag

        # dynamic features existence depending on model or plots
        self.features = {}

        # cache to save builder outputs to prevent recomputing
        self.cache = {}

    def has(self, key: str) -> bool:
        if key in self.features:
            return True

        if key in self.cache:
            return True

        if key.startswith("evaluation_state."):
            return self._has_nested(
                self.evaluation_state,
                key.split(".")[1:]
            )
        return False

    def _has_nested(self, obj, parts):
        for part in parts:
            if not isinstance(obj, dict) or part not in obj:
                return False
            obj = obj[part]
        return True

    def require(self, *keys: str):
        missing = [k for k in keys if k not in self.features]
        if missing:
            raise KeyError(f"Missing required optional features: {missing}")

    def get(self, key: str):
        if key in self.cache:
            return self.cache[key]

        if key in self.features:
            return self.features[key]

        if key.startswith("evaluation_state."):
            return self._get_nested(
                self.evaluation_state,
                key.split(".")[1:]
            )
        raise KeyError(key)

    def _get_nested(self, obj, parts):
        for part in parts:
            obj = obj[part]
        return obj

    def add_feature(self, name, feature):
        if torch.is_tensor(feature):
            feature = feature.detach().cpu()
        elif isinstance(feature, (list, tuple)):
            feature = [f.cpu() for f in feature]
        self.features[name] = feature

    def add_features(self, *named_features: tuple[str, torch.Tensor]):
        for name, feature in named_features:
            self.add_feature(name, feature)

    def add_cache(self, name, feature):
        if torch.is_tensor(feature):
            feature = feature.detach().cpu()
        self.cache[name] = feature

    def list_registered(self):
        return {
            "predictions", "targets", "target_map", "event_weights",
            *list(self.features.keys()),
            *list(self.cache.keys()),
            }

    def expand(self, pattern: str) -> list[str]:
        """
        Expand a glob pattern against all currently known keys.
        Returns a list of matching concrete keys.
        Example: 'monitor.gradients.*' -> ['monitor.gradients.dnn_score', 'monitor.gradients.binned_tensor', ...]
        Returns [pattern] unchanged if no wildcards present.
        """
        # no glob chars — no expansion needed
        if not any(c in pattern for c in ("*", "?", "[")):
            return [pattern]

        return [
            key for key in self.list_registered()
            if fnmatch.fnmatch(key, pattern)
        ]
