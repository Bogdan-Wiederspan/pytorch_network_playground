from bbTT.optimization.optimizer import init_optimizer
from bbTT.optimization.scheduler import SchedulerHandler, WarmupScheduler, init_scheduler
from bbTT.utils.lazy_loader import lazy_import

__all__ = ["SchedulerHandler", "WarmupScheduler", "init_optimizer", "init_scheduler"]


def __getattr__(name):
    return lazy_import(__name__, globals(), name)
