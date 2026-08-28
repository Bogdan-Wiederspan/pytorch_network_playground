from bbTT.monitoring.register import register_builder

# Simple template to create custom builder


# Rules:
# Name of Builder: needs to be unique
# Provides: Variable name in the return - this is registered in the global BUILD_REGISTRY
# Requires: Variable name resolved against BUILD_REGISTRY
@register_builder(
    "!NAME_OF_BUILDER",
    provides={
        "!VARIABLE_NAME",
    },
    requires={
        None
    },
)
def build_FUNCTION_NAME(ctx, **kwargs):
    CONSTRUCTED_VARIABLE = ctx.get("MY_VARIABLE")
    return {
        "!VARIABLE_NAME": CONSTRUCTED_VARIABLE,
    }
