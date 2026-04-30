from utils.lazy_loader import lazy_import

import src.train.export as export

def __getattr__(name):
    return lazy_import(__name__, globals(), name)
