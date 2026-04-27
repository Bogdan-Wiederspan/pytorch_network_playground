import torch

from data import features, load_data

# TODO save config in FILE at same place as Tensorboard
### data config
# targets and inputs
target_map = {"hh" : 0, "tt": 1, "dy": 2}
continuous_features, categorical_features = features.input_features(debug=False, debug_length=3)
dataset_pattern = [
    "dy_*","tt_*",
#   "hh_ggf_hbb_htt_kl0_kt1*",
    "hh_ggf_hbb_htt_kl1_kt1*"
    ]
eras = [
    "22pre",
    # "22post",
    # "23pre",
    # "23post"
    ]
datasets =  load_data.find_datasets(dataset_pattern, eras, file_type="root", verbose=False)
# changes in this dictionary will create a NEW hash of the data
dataset_config = {
    "continuous_features" : continuous_features,
    "categorical_features": categorical_features,
    "eras" : eras,
    "datasets" : datasets,
    "cuts" : None,
}

# Config for the model architecture
model_building_config = {
    # Rotation layer
    "enable_rotation": False, # enable rotation layer based on two reference objects, TODO: this breaks export to torch script
    "ref_phi_columns": ("res_dnn_pnet_vis_tau1", "res_dnn_pnet_vis_tau2"), # which column should be used to get rotation angle
    "rotate_columns": ("res_dnn_pnet_bjet1", "res_dnn_pnet_bjet2", "res_dnn_pnet_fatjet", "res_dnn_pnet_vis_tau1", "res_dnn_pnet_vis_tau2"), # which columns should be rotated
    # Padding Layers
    "categorical_padding_value": None, # missing values in categorical inputs are replaced by this
    "continuous_padding_value": None, # missing values in continuous inputs are replaced by this

    # General model
    "nodes": 128, # number of nodes in each linear layer of the dense blocks
    "activation_functions": "elu", # Marcel: elu
    "skip_connection_init": 1, # init value of the skip connection, 1 = exact copy
    "freeze_skip_connection": True, # True = non-learnable
    "batch_norm_eps" : 0.001, # Marcel: 0.001
    "LBN_M" : 10, # number of particles of the lbn network Marcel: 10
    "last_activation_fn" : "Softmax", # added activation function after last layer: Only Softmax, Sigmoid or None possible
}

config = {
    "model_choice": "bnet_lbn", # "binned_lbn" or "bnet_lbn"
    # "model_choice": "binned_lbn", # "binned_lbn" or "bnet_lbn"
    # "loss_fn" : "signal_efficiency",
    "loss_fn" : "signal_efficiency_binning_aware",

    "max_train_iteration" : 60000,
    "verbose_interval" : 5, # number of iterations between two logger outputs of training loss
    "validation_interval" : 100, # number of iterations between two validation passes / plots are done during validation
    "gamma":0.5,    "label_smoothing":0,
    "train_folds" : (0,), # which training folds to use
    "k_fold" : 5, # number of folds for k-fold cross validation
    "seed" : 100, # set torch and numpy seed for reproducibility
    "train_ratio" : 0.75, # split ratio for k-fold data into train and validation
    "v_batch_size" : -1, # for signal efficiency a validation pass needs to have whole WEIGHTs
    "t_batch_size" : 4096 * 10,
    "sample_ratio" : {"dy": 1/3, "tt": 1/3, "hh": 1/3},
    "sample_attributes" : (
        ["continuous", "categorical", "targets"] + ["product_of_weights", "evaluation_space_mask"]
        ), # what to sample from sampler - ATTENTION: continuous, categorical, targets needs to be always present
    "min_events_in_batch": 1, # minimal number of events of a subprocess in a batch
    "save_model_name" : "test_binning",

    "training_fn" : "default", # chooses the training function
    "validation_fn" : "signal_efficiency",

    "loss_mode" : "no_unc", # "full", "no_unc", "approximation" only relevant if signal_efficiency loss is chosen, determines which formula is used for the asimov calculation
    "loss_uncertainty" : 0.0, # only relevant if signal_efficiency loss is chosen, determines the background uncertainty used in the asimov calculation

    # DEBUG options
    "get_batch_statistic_return_dummy" : False,
    "load_marcel_stats" : False,
    "load_marcel_weights" : False,
}

# Config for the binning neural network, only relevant if model_choice is "binned_lbn"
# TODO since this is model relevant and new model hash should be created
binning_config = {
    # Binning Config
    # "init_edges": torch.linspace(.1,0.9, 25),
    "num_bins": 10,
    "lower_edge": 0,
    "upper_edge": 1,
    "binning_fn": torch.linspace, # EqualDistant, Decompress, or function defined in binning.__all__
    # Kernels Config
    "kernel_cls": "GaussianKernelV3",
    "kernel_config": {
        "GaussianKernelV3": {
            "smoothing_width": 0.3,
            "abs_mode": False,
            "left_notch": 0.1,
            "right_notch": 0.1,
            },
        },
    }

# Config for the learning rate scheduler
scheduler_config = {
    "patience" : 4, # marcel : 10
    "min_delta" : 0.00, # marcel : 0
    "threshold_mode" : "abs", # marcel : abs
    "factor" : 0.5,
}

# Config for the use optimizer, which is ADAMW
optimizer_config = {
    "apply_to": "weight",
    "decay_factor": 500,
    "normalize": True, # add weight normalized
    "lr":1e-3,
}
