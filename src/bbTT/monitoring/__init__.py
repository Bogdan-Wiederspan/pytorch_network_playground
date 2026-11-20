# Resolved lazily: the logger lives in this package and is imported by nearly everything
# (configs included), so importing it must not load the plots, builders and hooks (torch).
# ``load_registers`` is called explicitly by the training, not on import.
from bbTT.utils.lazy_loader import lazy_attributes

_ATTRIBUTE_MODULES = {
    "load_registers": "bootstrap",
    "EvalContext": "context.eval_context",
    "HookableModelMixin": "monitoring_hooks.hookable_model",
    "HookableMixin": "monitoring_hooks.hookable_module",
    "setup_monitoring": "monitoring_hooks.hookable_registration",
    "PlotContext": "register",
    "MONITOR_MODE_BATCH": "reporter",
    "MONITOR_MODE_TRAINING": "reporter",
    "MONITOR_MODE_VALIDATION": "reporter",
    "EvaluationReporter": "reporter",
    "EvaluationRunner": "runner",
    "TrainingMonitor": "training_monitor",
}

__all__ = list(_ATTRIBUTE_MODULES)
__getattr__ = lazy_attributes(__name__, globals(), _ATTRIBUTE_MODULES)
