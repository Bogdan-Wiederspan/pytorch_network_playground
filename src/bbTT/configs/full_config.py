from dataclasses import dataclass, field
from typing import Callable

from bbTT.configs import (
    BinningConfig,
    DataConfig,
    DebugConfig,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    RecordConfig,
    SamplerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from bbTT.monitoring.logger.logger import get_logger

logger_inst = get_logger("Configs")


@dataclass
class FullConfig:
    dataset_config: DataConfig = field(default_factory=DataConfig)
    model_building_config: ModelConfig = field(default_factory=ModelConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    binning_config: BinningConfig = field(default_factory=BinningConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss_config: LossConfig = field(default_factory=LossConfig)
    debug_config: DebugConfig = field(default_factory=DebugConfig)
    record_config: RecordConfig = field(default_factory=RecordConfig)
    sampler_config: SamplerConfig = field(default_factory=SamplerConfig)

    def _compatibility_rules(self) -> list[tuple[Callable[[], bool], str]]:
        """
        Each rule: (is_violated, message).
        When is_violated() returning True a raise happens.
        Thus all rules describe the FORBIDDEN state!
        """
        return [
            # add more rules here, needs to be a lambda function that returns a boolean
            # (lambda: BOOL1 and not BOOL2, "My Message")
        ]

    def __post_init__(self):
        num_violations = 0

        for rule, message in self._compatibility_rules():
            if rule():
                num_violations += 1
                logger_inst.warning(f"{num_violations}:{message}")

        if num_violations:
            raise ValueError("Any rule is broken, see log above and fix rule before continue.")
