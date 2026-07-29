from utils.lazy_loader import lazy_import

def __getattr__(name):
    return lazy_import(__name__, globals(), name)
