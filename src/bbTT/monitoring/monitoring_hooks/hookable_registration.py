from bbTT.monitoring.monitoring_hooks.hookable_model import HookableModelMixin
from bbTT.monitoring.monitoring_hooks.hookable_module import HookableMixin


def setup_monitoring(monitor, *hookable_objects):
    """
    Register a monitor against all hookable objects in this training run.
    Call once at setup — not per step.
    """
    for obj in hookable_objects:
        if isinstance(obj, (HookableMixin, HookableModelMixin)):
            obj.register_monitor(monitor)
