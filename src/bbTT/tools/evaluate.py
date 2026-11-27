"""
Run a trained checkpoint over its test, training or validation data and save the scores for the plots.

Usage: ``bbtt-evaluate --checkpoint_src my_run_fold0.pt --fold 0 --evaluate_on test --batch_size 4096 --save_dst my_run``.
"""
from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    """
    Entry point of ``bbtt-evaluate``; the evaluation logic stays in ``bbTT.evaluation.evaluate``.

    Args:
        argv (list[str] | None): Command line arguments without the program name.

    Returns:
        int: Exit code, 0 on success.
    """
    from bbTT.evaluation.evaluate import main as evaluate_main  # local import: torch, only needed to run

    return evaluate_main(argv)
