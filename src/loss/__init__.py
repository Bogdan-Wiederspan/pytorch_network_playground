from utils.lazy_loader import lazy_import

from loss import binning, kernel, loss_functions

def __getattr__(name):
    return lazy_import(__name__, globals(), name)
