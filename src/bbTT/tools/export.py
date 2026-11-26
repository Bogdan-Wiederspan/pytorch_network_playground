"""
Export a trained checkpoint to torch-export (.pt2) with a dynamic batch dimension, one file per fold.

Usage: ``bbtt-export --checkpoint_src my_run --fold 0,1,2`` (options: ``--head``, ``--add_activation``).
"""
from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    """
    Entry point of ``bbtt-export``; the export logic stays in ``bbTT.evaluation.export.pt2``.

    Args:
        argv (list[str] | None): Command line arguments without the program name.

    Returns:
        int: Exit code, 0 on success.
    """
    from bbTT.evaluation.export.pt2 import main as export_main  # local import: torch, only needed to run

    return export_main(argv)
