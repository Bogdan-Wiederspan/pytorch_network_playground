from bbTT.optimization.optimizer.factory import init_optimizer
from bbTT.utils.lazy_loader import lazy_import

__all__ = ["init_optimizer"]


def __getattr__(name):
    return lazy_import(__name__, globals(), name)
