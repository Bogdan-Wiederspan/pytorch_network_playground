from dataclasses import dataclass


@dataclass
class DebugConfig:
    get_batch_statistic_return_dummy: bool = False # do not calculate preprocessing, instead use mean = 0, std = 1
    load_marcel_stats: bool = False # load Marcels mean and std from preprocessing into STD Layer
    load_marcel_weights: bool = False # preload Marcels Pytorch Model weights, HINT: will break now, since architecture changes
