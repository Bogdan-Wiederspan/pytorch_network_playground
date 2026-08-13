from utils.lazy_loader import lazy_import

from optimizer import (SAM, utils, weight_decay, early_stopping, scheduler_handler)

def __getattr__(name):
    return lazy_import(__name__, globals(), name)
