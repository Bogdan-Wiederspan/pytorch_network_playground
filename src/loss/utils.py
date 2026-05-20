import torch

import loss

def init_loss(full_config, device=torch.device("cpu"), **kwargs):
    if full_config.training_config.loss_fn == "cross_entropy":
        # the default trainings_loss_function
        loss_cls = loss.loss_functions.WeightedCrossEntropy

        # train_loss_fn = torch.nn.CrossEntropyLoss(weight=None, size_average=None,label_smoothing=full_config.training_config.label_smoothing)
        # validation_loss_fn = torch.nn.CrossEntropyLoss(weight=None, size_average=None,label_smoothing=full_config.training_config.label_smoothing)
        train_loss_fn = loss_cls(weight=None, size_average=None,label_smoothing=full_config.training_config.label_smoothing)
        validation_loss_fn = loss_cls(weight=None, size_average=None,label_smoothing=full_config.training_config.label_smoothing)



    elif full_config.training_config.loss_fn == "signal_efficiency":
        training_sampler = kwargs["training_sampler"]
        train_loss_fn = loss.loss_functions.SignalEfficiency(
            sampler_inst=training_sampler,
            device=device,
            mode=full_config.training_config.loss_mode,
            uncertainty=full_config.training_config.loss_uncertainty,
            train=True,
            )
        validation_loss_fn = loss.loss_functions.SignalEfficiency(
            sampler_inst=training_sampler,
            device=device,
            mode=full_config.training_config.loss_mode,
            uncertainty=full_config.training_config.loss_uncertainty,
            train=False,
            )

    elif full_config.training_config.loss_fn == "signal_efficiency_binning_aware":

        training_sampler = kwargs["training_sampler"]
        bins = torch.linspace(full_config.binning_config.lower_edge, full_config.binning_config.upper_edge, full_config.binning_config.num_bins + 1)

        train_loss_fn = loss.loss_functions.BinningAwareSignificance(
            bins = bins,
            sampler_inst=training_sampler,
            device=device,
            mode=full_config.training_config.loss_mode,
            uncertainty=full_config.training_config.loss_uncertainty,
            binning_cfg=full_config.binning_config.kernel_config[full_config.binning_config.kernel_cls],
            train=True,
            )
        validation_loss_fn = loss.loss_functions.BinningAwareSignificance(
            bins = bins,
            sampler_inst=training_sampler,
            device=device,
            mode=full_config.training_config.loss_mode,
            uncertainty=full_config.training_config.loss_uncertainty,
            binning_cfg=full_config.binning_config.kernel_config[full_config.binning_config.kernel_cls],
            train=False,
        )

    return train_loss_fn, validation_loss_fn
