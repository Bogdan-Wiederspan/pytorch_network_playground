from utils.lazy_loader import lazy_import

from . import architectures, binning, blocks, input, preprocessing
from .register import MODEL_REGISTRY, register_model


def __getattr__(name):
    return lazy_import(__name__, globals(), name)
