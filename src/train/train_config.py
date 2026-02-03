from data.features import input_features
from data.load_data import find_datasets

### data config
# targets and inputs
target_map = {"hh" : 0, "tt": 1, "dy": 2}
continous_features, categorical_features = input_features(debug=False, debug_length=3)
dataset_pattern = ["dy_*","tt_*", "hh_ggf_hbb_htt_kl0_kt1*","hh_ggf_hbb_htt_kl1_kt1*"]
# eras = ["22pre", "22post", "23pre", "23post"]
eras = ["22pre"]
datasets =  find_datasets(dataset_pattern, eras, file_type="root", verbose=False)
# changes in this dictionary will create a NEW hash of the data
dataset_config = {
    "continous_features" : continous_features,
    "categorical_features": categorical_features,
    "eras" : eras,
    "datasets" : datasets,
    "cuts" : None,
}

# config of network
model_building_config = {
    "enable_rotation": False, # enbale rotiation layer based on two reference objects, TODO: this breaks export to torch script
    "ref_phi_columns": ("res_dnn_pnet_vis_tau1", "res_dnn_pnet_vis_tau2"), # which columsn should be used to get rotation angle
    "rotate_columns": ("res_dnn_pnet_bjet1", "res_dnn_pnet_bjet2", "res_dnn_pnet_fatjet", "res_dnn_pnet_vis_tau1", "res_dnn_pnet_vis_tau2"), # which columns should be rotated
    "categorical_padding_value": None, # missing values in categorical inputs are replaced by this
    "continous_padding_value": None, # missing values in continous inputs are replaced by this
    "nodes": 128, # number of nodes in each linear layer of the dense blocks
    "activation_functions": "elu",
    "skip_connection_init": 1, # init value of the skip connection, 1 = exact copy
    "freeze_skip_connection": True, # make skip connections static, non-learnable
    "batch_norm_eps" : 0.001, # marcel : 0.001
    "LBN_M" : 10, # number of particles of the lbn network

}

config = {
    "max_train_iteration" : 60000,
    "verbose_interval" : 25,
    "validation_interval" : 300,
    "gamma":0.5,
    "label_smoothing":0,
    "train_folds" : (0,),
    "k_fold" : 5,
    "seed" : 1,
    "train_ratio" : 0.75,
    "v_batch_size" : -1, # for signal efficiency a validation pass needs to have whole WEIGHTs
    "t_batch_size" : 4096,
    "sample_ratio" : {"dy": 1/3, "tt": 1/3, "hh": 1/3},
    "sample_attributes" : ["continous", "categorical", "targets"] + ["product_of_weights", "evaluation_space_mask"], # continous, categorical, targets should always be present
    "min_events_in_batch": 1,

    "save_model_name" : "test",
    "get_batch_statistic_return_dummy" : False,
    "load_marcel_stats" : False,
    "load_marcel_weights" : False,
    "training_fn" : "default", # chooses the training function
    "validation_fn" : "default",
    "loss_fn" : "signal_efficiency",
}

scheduler_config = {
    "patience" : 10, # marcel : 10
    "min_delta" : 0.01, # marcel : 0
    "threshold_mode" : "abs", # marcel : abs
    "factor" : 0.5,
}

optimizer_config = {
    "apply_to": "weight",
    "decay_factor": 500,
    "normalize": True, # add weight normalized
    "lr":1e-3,
}
