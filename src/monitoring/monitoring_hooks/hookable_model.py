from monitoring.monitoring_hooks.hookable_module import HookableMixin


class HookableModelMixin:
    """
    Model-level hook management. Aggregates hook interfaces from all
    HookableMixin sublayers. The loop only ever calls this interface —
    no knowledge of specific sublayer names or types required.
    """

    def _hookable_layers(self) -> list[HookableMixin]:
        """Returns all Layers that have hooking mechanism inplace."""
        return [
            module for module in self.modules()
            if isinstance(module, HookableMixin) and module is not self
        ]

    def register_monitor(self, monitor):
        """Wire monitor callbacks into every hookable sublayer. Call once at setup."""
        for layer in self._hookable_layers():
            layer.register_monitor(monitor)

    def enable_gradient_hooks(self):
        """Enable gradient monitoring for all layers"""
        for layer in self._hookable_layers():
            layer.enable_gradient_hooks()

    def disable_gradient_hooks(self):
        """Disable gradient monitoring for all layers"""
        for layer in self._hookable_layers():
            layer.disable_gradient_hooks()

    def enable_tensor_hooks(self):
        """Enable tensor monitoring for all layers"""
        for layer in self._hookable_layers():
            layer.enable_tensor_hooks()

    def disable_tensor_hooks(self):
        """Disable tensor monitoring for all layers"""
        for layer in self._hookable_layers():
            layer.disable_tensor_hooks()

    def monitored_gradient_names(self) -> list[str]:
        """Return list of all monitored gradients by name"""
        names = []
        for layer in self._hookable_layers():
            names.extend(layer.monitored_gradient_names())
        return names

    def monitored_tensor_names(self) -> list[str]:
        """Enable tensor monitoring for all layers"""
        names = []
        for layer in self._hookable_layers():
            names.extend(layer.monitored_tensor_names())
        return names
