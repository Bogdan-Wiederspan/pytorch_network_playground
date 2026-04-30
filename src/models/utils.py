
from models.create_model import BinnedLBNDenseNet, LBNDenseNet

def init_model(full_config):
    model_choice = full_config.training_config.model_choice

    if model_choice == "binned_lbn":
        model_inst = BinnedLBNDenseNet(full_config)
        model_inst.set_learning_mode("default")

    elif model_choice == "bnet_lbn":
        model_inst = LBNDenseNet(full_config)

    return model_inst
