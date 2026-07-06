from __future__ import annotations

import torch


class AddBinningLayer(torch.nn.Module):
    def __init__(
        self,
        model,
        binning_cls,
        kernel_cls,
        kernel_cfg,
        binning_edges,
        signal_cls=None,
        *args,
        **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.unbinned_model = model
        self.binning_layer = binning_cls(
            init_edges=binning_edges,
            kernel_cls=kernel_cls,
            kernel_cfg=kernel_cfg,
            )
        self.signal_cls = ... if signal_cls is None else signal_cls # used for class slicing

    def forward(self, categorical_inputs, continuous_inputs):
        x = self.unbinned_model(categorical_inputs, continuous_inputs)[:, self.signal_cls] # normal prediction as probabilities
        x = self.binning_layer(x)
        return x



class AddActFnToModel(torch.nn.Module):
    def __init__(self, model, act_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

        self.act_func = self._get_attr(torch.nn.modules.activation, act_fn)(dim=1)
        self.categorical_features = model.categorical_features
        self.continuous_features = model.continuous_features

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
