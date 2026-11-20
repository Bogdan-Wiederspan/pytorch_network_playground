# ``get_data`` is resolved lazily: importing a sibling such as ``find_datasets`` must not load
# the whole era pipeline (torch) along with it.
from bbTT.utils.lazy_loader import lazy_attributes

__all__ = ["get_data"]
__getattr__ = lazy_attributes(__name__, globals(), {"get_data": "era_pipeline"})
