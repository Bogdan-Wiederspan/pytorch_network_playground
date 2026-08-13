from utils.lazy_loader import lazy_import

def __getattr__(name):
    return lazy_import(__name__, globals(), name)

from .utils import EMPTY_INT, EMPTY_FLOAT  # re-export constants
from .logger import get_logger

__all__ = ["EMPTY_INT", "EMPTY_FLOAT", "get_logger"]
