import torch

from utils import logger

logger_inst = logger.get_logger(__name__)

class TrainingMonitor:
    def __init__(self, to_cpu:bool=True, non_blocking:bool=True):
        """
        Pure data store for gradients and tensor hooks.
        Tags every captures value with the step

        Args:
            to_cpu (bool, optional): Moves storage to CPU. Defaults to True.
            non_blocking (bool, optional): Does not block GPU when moving to CPU. If True backwards pass is always disrupted when moving. Defaults to True.
        """
        self.to_cpu=to_cpu
        self.non_blocking = non_blocking
        self.gradients = {} # name -> (step, tensor)
        self.tensors = {} # name -> (step, tensor)
        self._step = 0

    def start_step(self, step=None):
        # counter to measure staleness
        self._step = step if step is not None else self._step + 1

    def _capture(self, store, name, value):
        value = value.detach()
        if self.to_cpu:
            # non blocking prevents synchronization during backwards
            value = value.to("cpu", non_blocking = self.non_blocking)
        store[name] = (self._step, value)


    def state(self, unwrap:bool=True) -> dict[torch.Tensor, torch.Tensor]:
        """
        Copies of

        Args:
            unwrap (bool): Removes step value from gradients or tensor storage.

        Returns:
            dict[str, [torch.Tensor, torch.Tensor]]: Dict with gradients and tensor
        """
        if unwrap:
            return {
                "gradients" : {name: gradient for name, (step, gradient) in self.gradients.items()},
                "tensors" : {name: tensor for name, (step, tensor) in self.tensors.items()}
            }
        return {
            "gradients": self.gradients,
            "tensors": self.tensors
        }

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
        return [
            name for name in expected_names
            if name not in self.gradients or self.gradients[name][0] != self._step
        ]

    def check_gradient_correctness(self, expected_names):
        missing = self.stale_gradients(expected_names=expected_names)
        if missing:
            logger_inst.warning(f"Gradient hooks did not fire for {missing}")


    def clear_gradients(self):
        self.gradients.clear()

    def clear_tensors(self):
        self.tensors.clear()

    def clear(self):
        self.gradients.clear()
        self.tensors.clear()
