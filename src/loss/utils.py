

import torch

from loss import BinningAwareSignificance, SignalEfficiency, WeightedCrossEntropy
from utils.utils import CPU_DEVICE


def init_loss(full_config, device=CPU_DEVICE, **kwargs):
    cfg = full_config.loss_config

    if cfg.loss_fn == "cross_entropy":
        # the default trainings_loss_function
        from IPython import embed; embed(header="MESSAGE Line 15 | File: utils.py")
        train_loss_fn = WeightedCrossEntropy(weight=None, size_average=None,label_smoothing=full_config.training_config.label_smoothing)
        validation_loss_fn = WeightedCrossEntropy(weight=None, size_average=None,label_smoothing=full_config.training_config.label_smoothing)

    elif cfg.loss_fn == "signal_efficiency":
        loss_cfg = {
            "sampler_inst": kwargs["training_sampler"],
            "device": device,
            "asimov_cfg": cfg.active_config,
        }

        train_loss_fn = SignalEfficiency(train=True, **loss_cfg)
        validation_loss_fn = SignalEfficiency(train=False, **loss_cfg)

    elif cfg.loss_fn == "signal_efficiency_binning_aware":
        # TODO check if works
        loss_cfg = {
            "bins": torch.linspace(full_config.binning_config.lower_edge, full_config.binning_config.upper_edge, full_config.binning_config.num_bins + 1),
            "sampler_inst": kwargs["training_sampler"],
            "device": device,
            "asimov_cfg": cfg.active_config,
            "binning_cfg": full_config.binning_config.kernel_config[full_config.binning_config.kernel_cls],
        }

        train_loss_fn = BinningAwareSignificance(device=device, train=True, **loss_cfg)
        validation_loss_fn = BinningAwareSignificance(device=device, train=False, **loss_cfg)

    return train_loss_fn, validation_loss_fn
