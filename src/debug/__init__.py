from utils.lazy_loader import lazy_import

from debug import (
    # dnn_debug,
    # marcel_weight_translation,
    model_building_test_area,
)

def __getattr__(name):
    return lazy_import(__name__, globals(), name)
