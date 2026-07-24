import torch


class HookableModule(torch.nn.Module):
    """
    Enables a registration mechanism for hooks. Each layer can then define what kind of
    tensors are interesting to be monitoringable. The actual connection happens then in the
    Training loop.

    Args:
        torch (_type_): _description_
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._gradient_callbacks = {}
        self._gradient_hooks_enabled = False

    def enable_gradient_hooks(self):
        self._gradient_hooks_enabled = True

    def disable_gradient_hooks(self):
        self._gradient_hooks_enabled = False

    def register_gradient_callback(self, name, callback):
        self._gradient_callbacks[name] = callback

    def attach_gradient_hook(self, tensor, name):
        if self._gradient_hooks_enabled:
            callback = self._gradient_callbacks.get(name)
            if callback is not None:
                tensor.register_hook(callback)
