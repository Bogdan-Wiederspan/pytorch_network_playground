from bbTT.models import architectures, binning, blocks, input, preprocessing
from bbTT.models.register import MODEL_REGISTRY, register_model
from bbTT.utils.lazy_loader import lazy_import


def __getattr__(name):
    return lazy_import(__name__, globals(), name)
