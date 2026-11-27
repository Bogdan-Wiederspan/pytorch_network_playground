"""
Train the DNN with the configuration in ``bbTT/configs``, optionally resuming a checkpoint with overrides.

Usage: ``bbtt-train -tn my_run -sn my_run`` or ``bbtt-train -sn scan_lr -o optimizer_config.lr=3e-3`` (``--list-overrides`` shows all paths).
"""
from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    """
    Entry point of ``bbtt-train``; the training logic stays in ``bbTT.train.train``.

    Args:
        argv (list[str] | None): Command line arguments without the program name.

    Returns:
        int: Exit code, 0 on success.
    """
    from bbTT.train.train import main as train_main  # local import: torch, only needed to run

    return train_main(argv)
