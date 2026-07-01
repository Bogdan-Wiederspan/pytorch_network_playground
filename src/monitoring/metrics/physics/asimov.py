import statistics.asimov as _asimov
from functools import wraps

import torch


# provide a non gradient version of asimov for metrics
def _no_grad_asimov(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return fn(*args, **kwargs)
    return wrapper

asimov_metric = _no_grad_asimov(_asimov.asimov)
asimov_no_background_metric = _no_grad_asimov(_asimov.asimov_no_background)
asimov_small_signal_and_no_background_metric = _no_grad_asimov(_asimov.asimov_small_signal_and_no_background)
