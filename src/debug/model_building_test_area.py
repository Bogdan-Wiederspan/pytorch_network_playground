from __future__ import annotations

# package imports
import optimizer
import models.create_model as create_model

#from models import create_model
from data import load_data, preprocessing, sampler, cache
from utils import logger

from train.train_config import full_config




from utils.parser import ParserBuilder

def main(parser_args):
    logger_inst = logger.get_logger(__name__)

    # events = load_data.get_data(full_config.dataset_config, ignore_cache=parser_args.ignore_cache, _save_cache=parser_args.save_cache)


    full_config.model_building_config.mean, full_config.model_building_config.std = preprocessing.get_batch_statistics_from_sampler(
        features=full_config.dataset_config.continuous_features,
        return_dummy=True,
    )

    if full_config.training_config.model_choice == "binned_lbn":
        model_inst = create_model.BinnedLBNDenseNet(full_config)
        model_inst.set_learning_mode("default")
    elif full_config.training_config.model_choice == "bnet_lbn":
        model_inst = create_model.LBNDenseNet(full_config)
    else:
        raise ValueError(f"Model choice {full_config.training_config.model_choice} not recognized, but must be set")

    # only linear layers contribute to weight decay, prepare config that separates them for the optimizer
    weight_decay_parameters = optimizer.weight_decay.normalized_weight_decay(
            model_inst,
            full_config.optimizer_config.decay_factor,
            full_config.optimizer_config.normalize,
            )


if __name__ == "__main__":
    parser = ParserBuilder("cache")
    args = parser.args
    main(parser.args)
