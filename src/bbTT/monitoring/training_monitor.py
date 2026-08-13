from __future__ import annotations

import torch

from bbTT.utils import logger

logger_inst = logger.get_logger(__name__)

class TrainingMonitor:
    def __init__(self, to_cpu:bool=True, non_blocking:bool=True):
        """
        Data store for gradients and tensor hooks.
        A Monitor instance needs to be connected to the Torch Modules that has gradient monitoring enabled.

        Args:
            to_cpu (bool, optional): Moves storage to CPU. Defaults to True.
            non_blocking (bool, optional): Does not block GPU when moving to CPU. If True backwards pass is always disrupted when moving. Defaults to True.
        """
        self.to_cpu=to_cpu
        self.non_blocking = non_blocking
        self.gradients: dict[str, dict[str, tuple]] = {} # name -> (step, tensor)
        self.tensors: dict[str, dict[str, tuple]] = {} # name -> (step, tensor)
        self._steps: dict[str, float] = {} # type of loop -> (step)
        self._active_mode = None

    def set_mode(self, mode:str):
        self._active_mode = mode
        if mode not in self.gradients:
            self.gradients[mode] = {}

        if mode not in self.tensors:
            self.tensors[mode] = {}

        if mode not in self._steps:
            self._steps[mode] = 0

    def start_step(self, step=None):
        if self._active_mode is None:
            raise RuntimeError("Call set_mode( before start_step())")
        current_step = self.current_step()
        self._steps[self._active_mode] = current_step + 1

    def current_step(self, mode=None) -> int:
        mode = mode or self._active_mode
        return self._steps.get(mode, 0)

    def _capture(self, store, name, value):
        if self._active_mode is None:
            raise RuntimeError("Call set_mode() before capturing any tensor or gradient")
        value = value.detach()
        if self.to_cpu:
            # non blocking prevents synchronization during backwards
            value = value.to("cpu", non_blocking = self.non_blocking)
        store[self._active_mode][name] = (self.current_step(), value)


    def state(self, mode: str | None = None) -> dict[torch.Tensor, torch.Tensor]:
        """
        Snapshot for given mode or all if mode is None.

        Args:
            unwrap (bool): Removes step value from gradients or tensor storage.

        Returns:
            dict[str, [torch.Tensor, torch.Tensor]]: Dict with gradients and tensor
        """
        if mode is not None:
            return {
                "gradients" : self.get_gradients(mode),
                "tensors" : self.get_tensors(mode),
            }

        return {
            mode: {
                "gradients" : self.get_gradients(mode),
                "tensors" : self.get_tensors(mode),
            }
            for mode in set(self.gradients) | set(self.tensors)
        }

    def get_gradients(self, mode:str, unwrap=True, prefix=""):
        bucket = self.gradients.get(mode, {})
        if unwrap:
            return {f"{prefix}{name}": gradient for name, (step, gradient) in bucket.items()}
        return bucket

    def get_tensors(self, mode:str, unwrap=True, prefix=""):
        bucket = self.tensors.get(mode, {})
        if unwrap:
            return {f"{prefix}{name}": value for name, (step, value) in bucket.items()}
        return bucket

    def get_plot_gradients(self, mode):
        return tuple(self.get_gradients(mode, unwrap=True, prefix="monitored_gradient.").items())

    def get_plot_tensors(self, mode):
        return tuple(self.get_tensors(mode, unwrap=True, prefix="monitored_tensor.").items())


    def sync(self):
        """
        Force any non-blocking transfer to be done before continue working on ex. logging.
        Will do nothing when to_cpu=False OR non_blocking=False, or if source wasn't pinned memory
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def gradient_callback(self, name):
        def save(grad):
            self._capture(store=self.gradients, name=name, value=grad)
        return save

    def tensor_callback(self, name):
        def save(value):
            self._capture(store=self.tensors, name=name, value=value)
        return save

    def stale_gradients(self, expected_names):
        """Names that were expected this step but weren't captured, or
        are tagged with an older step (hook silently didn't fire)."""
        mode = self._active_mode
        if mode is None:
            return expected_names

        return [
            name for name in expected_names
            if name not in self.gradients.get(mode, {})
            or self.gradients[mode][name][0] != self.current_step()
        ]

    def check_gradient_correctness(self, expected_names):
        missing = self.stale_gradients(expected_names=expected_names)
        if missing:
            logger_inst.warning(f"Gradient hooks did not fire for {missing}")


    def clear_gradients(self):
        self.gradients.clear()

    def clear_tensors(self):
        self.tensors.clear()

    def clear(self, mode: str):
        # clear specific mode or all
        if mode is not None:
            self.gradients.get(mode, {}).clear()
            self.tensors.get(mode, {}).clear()
        else:
            self.gradients.clear()
            self.tensors.clear()
