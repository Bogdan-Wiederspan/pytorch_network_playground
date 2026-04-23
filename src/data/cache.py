import dataclasses
import hashlib
import os
import pathlib
import pickle

from utils.logger import get_logger

logger_inst = get_logger(__name__)

def hash_dictionary(config):
    hashable_dict = sorted(config.items(), key=lambda item: item[0])
    h = tuple(hashable_dict)
    h = hashlib.sha256(str(h).encode("utf-8")).hexdigest()[:10]
    return h

class DataCacher():
    def __init__(self, config):
        if dataclasses.is_dataclass(config):
            config = dataclasses.asdict(config)
        self.path = self.cache_dir(config)

    def cache_dir(self, config):
        h = hash_dictionary(config)
        p = pathlib.Path(os.environ["CACHE_DIR"])

        if not p.exists():
            raise FileExistsError("Cache dir does not exist")
        return p / h

    def create_cache_dir(self):
        self.path.mkdir(parents=False, exist=False)

    def save_cache(self, data):
        logger_inst.i_info(f"Try saving cache at {self.path}:")
        with open(f"{self.path}", "wb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        logger_inst.info(f"Done saving cache")

    def load_cache(self):
        if not self.path.exists():
            raise FileExistsError(f"Cache path {self.path} does not exist")

        logger_inst.i_info(f"Loading cache from {self.path}")
        with open(self.path, "rb") as file:
            events = pickle.load(file)
        logger_inst.info(f"Done loading cache")
        return events
