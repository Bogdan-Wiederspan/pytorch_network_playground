from __future__ import annotations

import torch

# package imports
import bbTT.models.architectures as arc

#from models.architectures import *
from bbTT.configs.full_config import FullConfig

#from models import create_model
from bbTT.data_handling import io, preprocessing
from bbTT.monitoring.logger.logger import get_logger


def main(**kwargs):
    full_config = FullConfig(**kwargs)
    model_inst = arc.BinnedLBNDenseNet(full_config)
    logger_inst = get_logger(__name__)

    # load test data
    events = io.get_data(full_config.dataset_config, rebuild_cache=False, _save_cache=False)
    hh = events[('hh', 21101)]

    full_config.model_building_config.mean, full_config.model_building_config.std = preprocessing.get_batch_statistics_from_sampler(
        features=full_config.dataset_config.continuous_features,
        return_dummy=True,
    )

    data = torch.load("/data/dust/user/wiedersb/HH_DNN/cache/test.pt", map_location="cpu")
    prediction, truth, weights = data["pred"], data["truth"], data["weights"]
    physics_parameters = None
    if model_inst.has_physics_parameters:
        # all events evaluated at kappa lambda = 1
        physics_parameters = torch.ones(len(hh["continuous"]), model_inst.num_physics_parameters)
    model_inst(hh["categorical"], hh["continuous"], physics_parameters)
    from IPython import embed; embed(header="MESSAGE Line 30 | File: model_building_test_area.py")



if __name__ == "__main__":
    main()
