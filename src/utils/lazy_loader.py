from __future__ import annotations

import importlib
import types
import sys

from importlib.util import find_spec

def lazy_import(package_name, globals_dict, name):
    try:
        module = importlib.import_module(f"{package_name}.{name}")
        globals_dict[name] = module  # cache
        return module
    except ModuleNotFoundError:
        raise AttributeError(f"module {package_name} has no attribute {name}")


def maybe_import(module_name):
    # check if actual module is there, if so return it
    if find_spec(module_name) is not None:
        return __import__(module_name)

    class DummyModule(types.ModuleType):
        def __getattr(self, name):
            def _dummy(*args, **kwargs):
                print(f"Dummy Module implemented return None instead of calling {name}")
                return None
            return _dummy
    dummy = DummyModule(module_name)
    # register as sys.modules, ensuring that normal import still works
    sys.modules[module_name] = dummy
    return dummy
