import os
import pathlib
import pickle
from typing import TYPE_CHECKING

from bbTT.monitoring.logger.logger import get_logger

logger_inst = get_logger(__name__)

if TYPE_CHECKING:
    from bbTT.configs.io_config import DataConfig


class DataCacher():
    def __init__(self, config: DataConfig):
        """
        Defines a cache saved in location defined by CACHE_DIR variable.
        All data stored under a hash that is created by given *config* dataclass.
        The hash defines a unique set of data, where same conditions are applied.
        The data is stored per ERA as pickle file.

        Args:
            config (DataConfig): DataConfig instance.
        """

        self.config = config

        self.hash = self.config.content_hash()
        self.cache_root = pathlib.Path(os.environ["CACHE_DIR"])

        self.path = self.cache_root / self.hash
        if not self.cache_root.exists():
            raise FileExistsError(f"Root directory does not exist at {self.cache_root} - create it" )
        self.path.mkdir(parents=False, exist_ok=True) # no automatic dir creation, want to prevent wrong paths


    def _era_path(self, era):
        return self.path / f"{era}.pkl"


    def era_exists(self, era: str):
        return self._era_path(era).exists()


    def save_era(self, era, events: dict):
        path = self._era_path(era)
        # save as tmp first, rename after successful save to prevent saving of corrupt files
        tmp = path.with_suffix(".tmp")
        logger_inst.debug(f"Saving: {era} --> {path}:")
        with open(tmp, "wb") as f:
            pickle.dump(events, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.rename(path)


    def load_era(self, era):
        path = self._era_path(era)
        if not path.exists():
            raise FileNotFoundError(f"Request Cache for era: {era} does not exit at {self.path}")

        logger_inst.debug(f"Loading cache from: {path}")
        with open(path, "rb") as file:
            return pickle.load(file)
