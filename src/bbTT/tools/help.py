"""
List the bbtt-* command line tools with a short description and an example call.

Usage: ``bbtt-help`` or ``bbtt-help sum-sizes`` for the full docstring of one tool.
"""
from __future__ import annotations

import argparse
import ast
import pathlib
import textwrap

TOOLS_DIR = pathlib.Path(__file__).parent


def tool_docstrings() -> dict[str, str]:
    """
    Read the module docstring of every tool without importing it (imports may pull in torch).
    A tool is a module in this package that defines ``main``; helper modules (``cli``) are skipped.

    Returns:
        dict[str, str]: Command name (``bbtt-sum-sizes``) -> module docstring, sorted by name.
    """
    docs = {}
    for path in sorted(TOOLS_DIR.glob("[!_]*.py")):
        module = ast.parse(path.read_text())
        if not any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in module.body):
            continue
        command = "bbtt-" + path.stem.replace("_", "-")
        docs[command] = ast.get_docstring(module) or "(no docstring)"

    return docs


def summary(docstring: str) -> tuple[str, str]:
    """
    Split a tool docstring into its one-line description and its ``Usage:`` example.

    Args:
        docstring (str): Module docstring following the convention in ``tools/__init__.py``.

    Returns:
        tuple[str, str]: ``(description, usage)``, inline code markers removed; usage empty if absent.
    """
    lines = [line.strip().replace("``", "") for line in docstring.splitlines()]
    description = lines[0] if lines else ""
    usage = next((line.partition(":")[2].strip() for line in lines if line.startswith("Usage:")), "")
    return description, usage


def main(argv: list[str] | None = None) -> int:
    """
    Entry point of ``bbtt-help``.

    Args:
        argv (list[str] | None): Command line arguments without the program name.

    Returns:
        int: Exit code, 0 on success, 1 for an unknown tool.
    """
    parser = argparse.ArgumentParser(prog="bbtt-help", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tool", nargs="?", help="tool name with or without the bbtt- prefix, shows its full docstring")
    args = parser.parse_args(argv)

    docs = tool_docstrings()
    if args.tool is not None:
        command = args.tool if args.tool.startswith("bbtt-") else f"bbtt-{args.tool}"
        if command not in docs:
            parser.error(f"unknown tool '{command}', known: {', '.join(docs)}")
        print(f"{command}\n\n{docs[command]}")
        return 0

    width = max(len(command) for command in docs)
    for command, docstring in docs.items():
        description, usage = summary(docstring)
        print(f"{command.ljust(width)}  {description}")
        if usage:
            print(textwrap.indent(f"example: {usage}", " " * (width + 2)))
    return 0
