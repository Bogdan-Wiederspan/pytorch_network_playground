from __future__ import annotations

import importlib
import sys
import types
from importlib.util import find_spec


def lazy_import(package_name, globals_dict, name):
    try:
        module = importlib.import_module(f"{package_name}.{name}")
        globals_dict[name] = module  # cache
        return module
    except ModuleNotFoundError as exception:
        raise AttributeError(f"{exception}\n module {package_name} has no attribute {name}")


def lazy_attributes(package_name: str, globals_dict: dict, attribute_modules: dict[str, str]):
    """
    Build a module level ``__getattr__`` that imports attributes from submodules on first access.

    Lets a package ``__init__`` offer ``from package import Name`` without importing every
    submodule up front, so a consumer of one light submodule does not pay for the heavy ones.

    Args:
        package_name (str): ``__name__`` of the package.
        globals_dict (dict): ``globals()`` of the package, used to cache resolved attributes.
        attribute_modules (dict[str, str]): Attribute name -> submodule (relative to the package) defining it.

    Returns:
        Callable[[str], Any]: The ``__getattr__`` to assign in the package.
    """

    def __getattr__(name: str):
        if name not in attribute_modules:
            raise AttributeError(f"module {package_name} has no attribute {name}")
        value = getattr(importlib.import_module(f"{package_name}.{attribute_modules[name]}"), name)
        globals_dict[name] = value  # cache, __getattr__ is not called again for this name
        return value

    return __getattr__


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
