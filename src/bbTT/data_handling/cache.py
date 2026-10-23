from __future__ import annotations

import json
import os
import pathlib
from typing import TYPE_CHECKING

import torch

from bbTT.monitoring.logger.logger import get_logger

logger_inst = get_logger(__name__)

if TYPE_CHECKING:
    from bbTT.configs.io_config import DataConfig


class BaseCacher:
    def __init__(self, config):
        self.config = config
        self.hash = self.config.content_hash()
        self.cache_root = pathlib.Path(os.environ["CACHE_DIR"])

        self.path = self.cache_root / self.hash
        if not self.cache_root.exists():
            raise FileExistsError(f"Root directory does not exist at {self.cache_root} - create it")
        self.path.mkdir(parents=False, exist_ok=True)  # no automatic dir creation, want to prevent wrong paths


class DataCacher(BaseCacher):
    def __init__(self, config: DataConfig):
        """
        Defines a cache saved in location defined by CACHE_DIR variable.
        All data stored under a hash that is created by given *config* dataclass.
        The hash defines a unique set of data, where same conditions are applied.
        The data is stored per ERA as torch .pt file.

        Args:
            config (DataConfig): DataConfig instance.
        """
        super().__init__(config=config)

    def _era_path(self, era) -> pathlib.Path:
        return self.path / f"{era}.pt"

    def era_exists(self, era: str) -> bool:
        return self._era_path(era).exists()

    def save_era(self, era, events: dict[tuple[str, int], dict[str, torch.Tensor]]) -> None:
        """
        Persist one era with ``torch.save``.
        Tensors that are views of the same storage (e.g. the mask columns) are stored once.

        Args:
            era (str): Era name, used as file stem.
            events (dict[tuple[str, int], dict[str, torch.Tensor]]): Events of this era, keyed by uid.
        """
        path = self._era_path(era)
        # save as tmp first, rename after successful save to prevent saving of corrupt files
        tmp = path.with_suffix(".tmp")
        logger_inst.debug(f"Saving: {era} --> {path}:")
        torch.save(events, tmp)
        tmp.rename(path)

    def load_era(self, era):
        """
        Memory-map one era. Nothing is read into RAM here: every tensor is backed by the cache
        file, pages are loaded on first access and can be dropped again by the OS.
        The mapping is copy-on-write, so the cache file is never modified through the returned tensors.

        Args:
            era (str): Era name.

        Returns:
            dict[tuple[str, int], dict[str, torch.Tensor]]: Same structure as saved, tensors file-backed.

        Raises:
            FileNotFoundError: If the era was never cached.
        """
        path = self._era_path(era)
        if not path.exists():
            raise FileNotFoundError(f"Request Cache for era: {era} does not exit at {self.path}")

        logger_inst.debug(f"Memory-mapping cache from: {path}")
        # load a handle and access underlying array from disk when needed
        return torch.load(path, mmap=True, weights_only=True)

    def _sizes_path(self, era, is_pid=True) -> pathlib.Path:
        if is_pid:
            return self.path / f"{era}_pid_sizes.json"
        return self.path / f"{era}_dataset_sizes.json"

    def save_sizes(self, era, sizes: dict, is_pid=True):
        path = self._sizes_path(era=era, is_pid=is_pid)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(sizes, f, indent=2)
        tmp.rename(path)

    def load_era_sizes(self, era, is_pid=True) -> dict:
        path = self._sizes_path(era, is_pid=is_pid)
        if not path.exists():
            return {}
        with open(path, "r") as f:
            return json.load(f)
