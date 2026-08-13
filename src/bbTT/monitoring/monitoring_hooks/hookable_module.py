from __future__ import annotations

from contextlib import contextmanager

from bbTT.utils import logger

logger_inst = logger.get_logger(__name__)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from monitoring.training_monitor import TrainingMonitor

class HookableMixin:
    """
    Enables a registration mechanism for hooks.
    Hooks can be either for backward or forward pass.
    During backward gradients are monitored, during forward the actual tensor.
    In the forward of a Module, one can define which tensors should be monitored.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gradient_callbacks = {}
        self.tensor_callbacks = {}
        self._gradient_hook_handles = {}
        self._gradient_hooks_enabled = False
        self._tensor_hooks_enabled = True

    def monitored_gradient_names(self) -> list[str]:
        """
        Names of all Gradients that are exposed by the Layer for monitoring.

        Returns:
            list[str]: List of all monitored names.
        """
        return []

    def monitored_tensor_names(self) ->list[str]:
        """
        Names of all Tensors that are exposed by the Layer for monitoring.

        Returns:
            list[str]: List of all monitored names.
        """
        return []

    def register_monitor(self, monitor: TrainingMonitor):
        """
        Connects external monitor callback functions with monitored names.

        Args:
            monitor (TrainingMonitor): TrainingMonitor instance that stores all gradients and tensors.
        """
        # TODO docstring
        for name in self.monitored_gradient_names():
            self.register_gradient_callback(
                name=name,
                callback=monitor.gradient_callback(name)
                )
        for name in self.monitored_tensor_names():
            self.register_tensor_callback(
                name=name,
                callback=monitor.tensor_callback(name)
                )

    # --- registration of callbacks, should be called 1x per setup ---

    def register_gradient_callback(self, name, callback):
        if name in self.gradient_callbacks and self.gradient_callbacks[name] is not callback:
            logger_inst.warning(
                f"Overwriting gradient callback for '{name}' with a different callback."
                )

        self.gradient_callbacks[name] = callback

    def register_tensor_callback(self, name, callback):
        if name in self.tensor_callbacks and self.tensor_callbacks[name] is not callback:
            logger_inst.warning(
                f"Overwriting tensor callback for '{name}' with a different callback."
                )

        self.tensor_callbacks[name] = callback

    def unregister_gradient_callback(self, name):
        self.gradient_callbacks.pop(name, None)
        self._remove_gradient_hook(name)

    def unregister_tensor_callback(self, name):
        self.tensor_callbacks.pop(name, None)

    # --- enable / disable commands ---
    def enable_gradient_hooks(self):
        self._gradient_hooks_enabled = True

    def disable_gradient_hooks(self):
        self._gradient_hooks_enabled = False

    def enable_tensor_hooks(self):
        self._tensor_hooks_enabled = True

    def disable_tensor_hooks(self):
        self._tensor_hooks_enabled = False

    @contextmanager
    def gradient_hooks_scope(self):
        self.enable_gradient_hooks()
        try:
            yield
        finally:
            self.disable_gradient_hooks()

    # --- attach, which is called from forward(), once per tensor and per pass

    def _remove_gradient_hook(self, name):
        handle = self._gradient_hook_handles.pop(name, None)
        if handle is not None:
            handle.remove()

    def remove_gradient_hooks(self):
        for handle in self._gradient_hook_handles.values():
            handle.remove()
        self._gradient_hook_handles.clear()

    def monitor_gradient(self, tensor, name):
        # when no gradient required OR no callback registered OR tensor no gradient has skip monitoring
        if not self._gradient_hooks_enabled:
            return

        callback = self.gradient_callbacks.get(name)
        if callback is None:
            return

        if not tensor.requires_grad:
            logger_inst.warning(f"Tensor {name} does not requires grad: Skipping")
            return

        # replace old hook with new one, throw warning if error comes from it and register handle
        self._remove_gradient_hook(name)

        def _safe_callback(grad):
            try:
                callback(grad)
            except Exception as e:
                logger_inst.warning(f"Gradient callback {name} raised {e}")
        self._gradient_hook_handles[name] = tensor.register_hook(_safe_callback)


    def monitor_tensor(self, tensor, name):
        if not self._tensor_hooks_enabled:
            return

        callback = self.tensor_callbacks.get(name)
        if callback is None:
            return
        try:
            callback(tensor)
        except Exception as e:
            logger_inst.warning(f"Tensor callback {name} raised {e}")
        callback(tensor)
