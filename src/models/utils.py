import torch
from models.layers import BinningLayer

from models.create_model import MODEL_REGISTRY

def init_model(full_config):
    model_choice = full_config.training_config.model_choice

    model_cls = MODEL_REGISTRY[model_choice]
    model_inst = model_cls(full_config)

    # extra settings for specific models
    if model_choice in ("lbn_dense", "binned_lbn_dense"):
        model_inst.set_learning_mode("model_only")

    return model_inst

class AddBinning(torch.nn.Module):
    def __init__(self, model, kernel_cls, kernel_cfg, binning_edges, signal_cls=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unbinned_model = model
        self.binning_layer = BinningLayer(
            init_edges=binning_edges,
            kernel_cls=kernel_cls,
            kernel_cfg=kernel_cfg,
            )
        self.signal_cls = ... if signal_cls is None else signal_cls # used for class slicing
        # for attr in self.model.mark_attributes():
        #     setattr(self, attr)

    def forward(self, categorical_inputs, continuous_inputs):
        x = self.unbinned_model(categorical_inputs, continuous_inputs)[:, self.signal_cls] # normal prediction as probabilities
        x = self.binning_layer(x)
        return x


class AddActFnToModel(torch.nn.Module):
    def __init__(self, model, act_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        # self.categorical_features = model.categorical_features
        # self.continuous_features = model.continuous_features

        self.act_func = self._get_attr(torch.nn.modules.activation, act_fn)(dim=1)
        self.categorical_features = model.categorical_features
        self.continuous_features = model.continuous_features
        # for attr in self.model.mark_attributes():
        #     setattr(self, attr)

    def _get_attr(self, obj, attr):
        for o in dir(obj):
            if o.lower() == attr.lower():
                return getattr(obj, o)
        else:
            raise AttributeError(f"Object has no attribute '{attr}'")

    def forward(self, categorical_inputs, continuous_inputs):
        x = self.model(categorical_inputs, continuous_inputs)
        x = self.act_func(x)
        return x
