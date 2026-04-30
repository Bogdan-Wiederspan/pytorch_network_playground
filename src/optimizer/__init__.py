from utils.lazy_loader import lazy_import

from optimizer import (SAM, utils, weight_decay)

def __getattr__(name):
    return lazy_import(__name__, globals(), name)
