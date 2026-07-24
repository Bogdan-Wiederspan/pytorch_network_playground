def inspect_gradients(module, prefix=""):
    for name, param in module.named_parameters():
        if param.requires_grad:
            status = (
                "HAS GRAD"
                if param.grad is not None
                else "NO GRAD"
            )

            print(
                f"{prefix}{name:60s} {status}"
            )
