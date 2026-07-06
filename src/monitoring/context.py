import torch


class EvalContext:
    def __init__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        target_map: dict[str, int],
        event_weights: torch.Tensor,
        ):
        """
        Context Object is a manager to hold everything relevant for evaluation and monitoring.
        Since this is an evaluation object no gradients are required.
        Due to plotting requirements, everything is moved to cpu.

        Args:
            predictions (torch.Tensor): Prediction of the model that is used to train the network
            class_predictions (torch.Tensor): Class prediction of the model, used for monitoring
            target (torch.Tensor): Truth value to be predicted
            target_map (torch.Tensor): Mapping of the targets to their corresponding node
            event_weights (torch.Tensor): Product of alle weights, unique per event
        """
        # core features, always exist
        self.predictions = predictions.detach().cpu()
        self.targets = targets.detach().cpu()
        self.target_map = target_map
        self.event_weights = event_weights.detach().cpu()

        # dynamic features existence depending on model or plots
        self.features = {}
        # cache to save builder outputs to prevent recomputing
        self.cache = {}

    def has(self, key: str) -> bool:
        flag = (
            key in self.features
            or
            key in self.cache
        )
        return flag

    def require(self, *keys: str):
        missing = [k for k in keys if k not in self.feature]
        if missing:
            raise KeyError(f"Missing required optional features: {missing}")

    def get(self, key: str):
        if key in self.cache:
            return self.cache[key]

        if key in self.features:
            return self.features[key]

        if key in self.scalar:
            return self.scalar[key]

        raise KeyError(key)

    def add_feature(self, name, feature):
        if torch.is_tensor(feature):
            feature = feature.detach().cpu()
        self.features[name] = feature

    def add_cache(self, name, feature):
        if torch.is_tensor(feature):
            feature = feature.detach().cpu()
        self.cache[name] = feature
