"""
Shared command line helpers for the bbtt-* tools.
"""
from __future__ import annotations

import argparse
import sys


def parse_args(parser: argparse.ArgumentParser, argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse ``argv`` like ``parser.parse_args``, but print the help and exit when no argument is given.

    Every tool explains itself first: running it bare never does anything. A tool whose
    defaults are useful offers an explicit flag for them (``bbtt-sum-sizes --current``).

    Args:
        parser (argparse.ArgumentParser): Parser of the tool.
        argv (list[str] | None): Arguments without the program name, ``None`` reads ``sys.argv``.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        parser.print_help()
        sys.exit(0)
    return parser.parse_args(argv)
