# contains tools to convert old model to new structure
# this is necessary due to the way pickle stores stuff
import sys
import types


def find_pickle_refs(path) -> set[tuple[str, str]]:
    """
    Look throught *path* and find all reference that pickle will look for WITHOUT executions

    Args:
        path (_type_): Path to pickle file.

    Returns:
        _type_: Set of tuples: (reference, type)
    """
    import pickletools
    import re
    import zipfile

    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        pkl_name = [n for n in names if n.endswith("data.pkl")][0]
        with z.open(pkl_name) as f:
            data = f.read()

    refs = set()
    out = []
    pickletools.dis(data, out=type("W", (), {"write": lambda self, s: out.append(s)})())
    text = "".join(out)

    for line in text.splitlines():
        if "GLOBAL" in line:
            m = re.search(r"GLOBAL\s+'([\w\.]+)\s+([\w\.]+)'", line)
            if m:
                refs.add((m.group(1), m.group(2)))
    return refs


class ClassMap:
    def __init__(self, map, from_version, destination_version):
        self.map = map
        self.current_version = from_version
        self.destination_version = destination_version

def install_fake_modules(class_map: dict[tuple[str, str], tuple[str, str]]) -> None:
    """Create fake modules at the OLD dotted paths, each populated with
    attributes pointing at the real objects from their NEW locations, and
    register them (and every parent package) in sys.modules."""

    # old_module_name -> {attr_name: real_object}
    grouped: dict[str, dict[str, object]] = {}

    for (old_mod, old_name), (new_mod, new_name) in class_map.items():
        real_obj = _import_attr(new_mod, new_name)
        grouped.setdefault(old_mod, {})[old_name] = real_obj

    for old_mod, attrs in grouped.items():
        fake = sys.modules.get(old_mod) or types.ModuleType(old_mod)
        for attr_name, obj in attrs.items():
            setattr(fake, attr_name, obj)
        sys.modules[old_mod] = fake
        _register_parent_packages(old_mod)


def _import_attr(module_name: str, attr_name: str) -> object:
    import importlib
    mod = importlib.import_module(module_name)
    return getattr(mod, attr_name)


def _register_parent_packages(dotted: str) -> None:
    """Make sure e.g. 'models' exists in sys.modules as a package if we
    only registered 'models.layers', so `import models.layers` style
    lookups during unpickling don't fail on the parent."""
    parts = dotted.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            pkg = types.ModuleType(parent)
            pkg.__path__ = []  # mark as a package
            sys.modules[parent] = pkg




map_v1_to_v2 = ClassMap(
    from_version=1,
    destination_version=2,
    map={
        ("models.create_model", "LBNDenseNet")       : ("bbTT.models.architectures.LBNDenseNet", "LBNDenseNet"),
        ("models.layers", "CatEmbeddingLayer")       : ("bbTT.models.preprocessing", "CatEmbeddingLayer"),
        ("models.layers", "CategoricalInputLayer")   : ("bbTT.models.input.categorical", "CategoricalInputLayer"),
        ("models.layers", "CategoricalTokenizer")    : ("bbTT.models.preprocessing.embedding", "CategoricalTokenizer"),
        ("models.layers", "ContinuousInputLayer")    : ("bbTT.models.input.continuous", "ContinuousInputLayer"),
        ("models.layers", "DenseBlock")              : ("bbTT.models.blocks", "DenseBlock"),
        ("models.layers", "DenseNetBlock")           : ("bbTT.models.blocks", "DenseNetBlock"),
        ("models.layers", "LBN")                     : ("bbTT.models.physics.lbn", "LBN"),
        ("models.layers", "LBNFeaturerExtractor")    : ("bbTT.models.physics.lbn_feature_extractor", "LBNFeatureExtractor"),
        ("models.layers", "LBN_DNN")                 : ("bbTT.models.physics.lbn_pipeline", "LBNPipeline"),
        ("models.layers", "OptionalInputLayer")      : ("bbTT.models.input", "OptionalInputLayer"),
        ("models.layers", "StandardizeLayer")        : ("bbTT.models.preprocessing", "StandardizeLayer"),
        ("models.layers", "WeightNormalizedLinear")  : ("bbTT.models.utils", "WeightNormalizedLinear"),
        ("train.train_config", "BinningConfig")      : ("bbTT.configs.binning_config", "BinningConfig"),
        ("train.train_config", "DataConfig")         : ("bbTT.configs.io_config", "DataConfig"),
        ("train.train_config", "ModelBuildingConfig"): ("bbTT.configs.model_config", "ModelConfig"),
}
)

install_fake_modules(map_v1_to_v2.map)
checkpoint_path = resolve_checkpoint_path(path)
checkpoint_inst = load_checkpoint(checkpoint_path)
full_config = rebuild_dataclass_from_dict(checkpoint_inst["full_config"])

from IPython import embed; embed(header="MESSAGE Line 109 | File: /afs/desy.de/user/w/wiedersb/xxl/pytorch_network_playground/src/bbTT/evaluation/conversion.py")
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "discover":
        discover(sys.argv[2])
    elif cmd == "convert":
        if len(sys.argv) < 4:
            print("usage: python pickle_remap.py convert <old_path> <new_path>")
            sys.exit(1)
        convert(sys.argv[2], sys.argv[3])
    else:
        print(f"unknown command: {cmd}")
        sys.exit(1)
