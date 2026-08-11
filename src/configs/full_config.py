from dataclasses import dataclass, field

from src.configs import (
    BinningConfig,
    DataConfig,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)


@dataclass
class FullConfig:
    dataset_config: DataConfig = field(default_factory=DataConfig)
    model_building_config: ModelConfig = field(default_factory=ModelConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    binning_config: BinningConfig = field(default_factory=BinningConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss_config: LossConfig = field(default_factory=LossConfig)

    def __post_init__(self):
        if (self.training_config.training_fn != "sam") and (self.optimizer_config.optimizer_choice == "sam"):
            raise ValueError(
                f"When using Optimizer SAM, training_fn must be 'sam', "
                f"currently: {self.training_config.training_fn}"
            )
