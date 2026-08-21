import dataclasses
import importlib
import inspect
import pkgutil
import typing


def from_dict(cls, data):
    if not dataclasses.is_dataclass(cls):
        return data

    hints = typing.get_type_hints(cls)
    # __new__ creates bar instance without calling init
    obj = object.__new__(cls)  # reconstruction of dataclasses reuse data class structure BUT dont use post_init anymore

    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue

        value = data[f.name]
        field_type = hints[f.name]
        origin = typing.get_origin(field_type)
        args = typing.get_args(field_type)

        if dataclasses.is_dataclass(field_type) and isinstance(value, dict):
            value = from_dict(field_type, value)
        elif origin in (list, tuple) and args and dataclasses.is_dataclass(args[0]) and isinstance(value, list):
            value = [from_dict(args[0], item) if isinstance(item, dict) else item for item in value]
            if origin is tuple:
                value = tuple(value)
        elif origin is tuple and isinstance(value, list):
            value = tuple(value)

        # instead of plain __setattr__ using object.__setatr___
        # matters only for frozen dataclasses, since here normal __setattr__ creates raises
        object.__setattr__(obj, f.name, value)
    return obj


def collect_dataclasses(package):
    """Import every submodule in `package` and collect all dataclasses defined there."""
    found = {}
    for _, modname, _ in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"{package.__name__}.{modname}")
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if dataclasses.is_dataclass(obj) and obj.__module__ == module.__name__:
                found[name] = obj
    return found


# ALL_DATACLASSES = collect_dataclasses(configs_pkg)
