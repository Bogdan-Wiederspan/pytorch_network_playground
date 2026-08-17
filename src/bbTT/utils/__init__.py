from .utils import EMPTY_FLOAT, EMPTY_INT  # re-export constants
from .lazy_loader import lazy_import



def __getattr__(name):
    return lazy_import(__name__, globals(), name)


__all__ = ["EMPTY_INT", "EMPTY_FLOAT"]
