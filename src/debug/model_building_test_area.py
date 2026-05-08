from __future__ import annotations

# package imports
import torch

import optimizer
import models.create_model as create_model
import models.layers as layers
#from models import create_model
from data import load_data, preprocessing, sampler, cache
from utils import logger
from optimizer.utils import init_optimizer
import loss.kernel as kernel
from train.train_config import full_config
from models.utils import init_model



from utils.parser import ParserBuilder

def main(parser_args):
    logger_inst = logger.get_logger(__name__)

    # events = load_data.get_data(full_config.dataset_config, ignore_cache=parser_args.ignore_cache, _save_cache=parser_args.save_cache)


    full_config.model_building_config.mean, full_config.model_building_config.std = preprocessing.get_batch_statistics_from_sampler(
        features=full_config.dataset_config.continuous_features,
        return_dummy=True,
    )


    data = torch.load("/data/dust/user/wiedersb/HH_DNN/cache/test.pt", map_location="cpu")
    prediction, truth, weights = data["pred"], data["truth"], data["weights"]

    binning_layer = layers.BinningLayerRight(
        num_bins=20,
        bounds=(0,1),
        binning_fn=torch.linspace,
        kernel_cls=kernel.GaussianKernelFinal,
        kernel_cfg={
            "left_notch" : 0,
            "right_notch" : 0,
            "smoothing_width" : 0.1,
            "abs_mode" : False,
            "bin_height" : 1,
        }
    )
    model_inst = init_model(full_config)



    from IPython import embed; embed(header="MESSAGE Line 30 | File: model_building_test_area.py")

    #### TEST OPTIMIZER

    # optimizer_inst = init_optimizer(full_config=full_config, model_inst=model_inst)

    # # only linear layers contribute to weight decay, prepare config that separates them for the optimizer
    # weight_decay_parameters = optimizer.weight_decay.normalized_weight_decay(
    #         model_inst,
    #         full_config.optimizer_config.decay_factor,
    #         full_config.optimizer_config.normalize,
    #         )


if __name__ == "__main__":
    parser = ParserBuilder("cache")
    args = parser.args
    main(parser.args)
