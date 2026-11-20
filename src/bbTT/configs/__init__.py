# The config classes are resolved lazily: ``from bbTT.configs import DataConfig`` loads only
# io_config, not the whole package. Some configs import torch (sampler_config, binning_config),
# and a config-only consumer (cache hash, command line tools) must not pay that start-up cost.
from bbTT.utils.lazy_loader import lazy_attributes

# class name -> module of this package that defines it
_CONFIG_MODULES = {
    "BinningConfig": "binning_config",
    "DebugConfig": "debug_config",
    "DataConfig": "io_config",
    "LossConfig": "loss_config",
    "ModelConfig": "model_config",
    "OptimizerConfig": "optimizer_config",
    "RecordConfig": "record_config",
    "SchedulerConfig": "scheduler_config",
    "TrainingConfig": "training_config",
    "SamplerConfig": "sampler_config",
}

__all__ = list(_CONFIG_MODULES)
__getattr__ = lazy_attributes(__name__, globals(), _CONFIG_MODULES)
