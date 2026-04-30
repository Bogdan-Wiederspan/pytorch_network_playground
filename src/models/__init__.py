from utils.lazy_loader import lazy_import

from models import create_model, layers, utils

def __getattr__(name):
    return lazy_import(__name__, globals(), name)
