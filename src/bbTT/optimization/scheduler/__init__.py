from bbTT.optimization.scheduler.factory import init_scheduler
from bbTT.optimization.scheduler.handler import SchedulerHandler
from bbTT.optimization.scheduler.warmup import WarmupScheduler
from bbTT.utils.lazy_loader import lazy_import

__all__ = ["SchedulerHandler", "WarmupScheduler", "init_scheduler"]


def __getattr__(name):
    return lazy_import(__name__, globals(), name)
