
"""
Sum the per-key counts of the ``<era>_{dataset,pid}_sizes.json`` files of a data cache into one table.

The cache lives in ``$CACHE_DIR/<hash>``, the hash defaults to the one of the current ``DataConfig``.

Usage: ``bbtt-sum-sizes --current`` or ``bbtt-sum-sizes [--pid] [--eras 22pre 22post] [--hash <hash>]``
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
from typing import get_args

import pandas as pd

from bbTT.configs.io_config import ERAS_CHOICE
from bbTT.tools.cli import parse_args

ERAS: tuple[str, ...] = get_args(ERAS_CHOICE)  # chronological order of the known eras


def find_sizes_files(cache_dir: pathlib.Path, kind: str, eras: list[str] | None) -> list[pathlib.Path]:
    """
    Collect the sizes files of one cache in chronological era order.

    Args:
        cache_dir (pathlib.Path): ``$CACHE_DIR/<hash>``.
        kind (str): ``dataset`` or ``pid``.
        eras (list[str] | None): Eras to include, ``None`` takes every era found in the cache.

    Returns:
        list[pathlib.Path]: Existing ``<era>_<kind>_sizes.json`` files (naming as in ``DataCacher._sizes_path``).

    Raises:
        FileNotFoundError: If no file is found or a requested era is missing.
    """
    if eras is None:
        eras = [era for era in ERAS if (cache_dir / f"{era}_{kind}_sizes.json").exists()]
    files = [cache_dir / f"{era}_{kind}_sizes.json" for era in sorted(eras, key=ERAS.index)]
    missing = [path for path in files if not path.exists()]
    if missing or not files:
        raise FileNotFoundError(f"no {kind} sizes found in {cache_dir}" + (f", missing: {missing}" if missing else ""))
    return files



def sum_sizes(files: list[pathlib.Path]) -> pd.DataFrame:
    """
    Build the summed table: one column per era, a total, grouped by category.

    The category is the first token of a dataset name (``tt_sl_powheg`` -> ``tt``); plain
    integer keys (process ids) all fall into one unnamed category.

    Args:
        files (list[pathlib.Path]): Sizes files named ``<era>_<kind>.json``, one per era.

    Returns:
        pd.DataFrame: Index (category, key), one column per era plus ``total``; sorted by
        category subtotal, then by total, descending. Missing keys count as 0.
    """
    # column name = era = file name up to the first "_"
    table = pd.DataFrame({path.name.partition("_")[0]: json.loads(path.read_text()) for path in files}).fillna(0).astype(int)
    table["total"] = table.sum(axis=1)
    table["category"] = ["" if key.isdigit() else key.partition("_")[0] for key in table.index]

    subtotal = table.groupby("category")["total"].transform("sum")
    table = table.assign(subtotal=subtotal).sort_values(["subtotal", "total"], ascending=False).drop(columns="subtotal")
    return table.rename_axis("key").set_index("category", append=True).swaplevel()


def main(argv: list[str] | None = None) -> int:
    """
    Entry point of ``bbtt-sum-sizes``.

    Args:
        argv (list[str] | None): Command line arguments without the program name.

    Returns:
        int: Exit code, 0 on success.
    """
    parser = argparse.ArgumentParser(prog="bbtt-sum-sizes", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pid", action="store_true", help="sum the process id sizes instead of the dataset sizes")
    parser.add_argument("--eras", nargs="+", choices=ERAS, help="eras to include, default: all found in the cache")

    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--current", action="store_true", help="the cache of the current DataConfig (the default, spelled out)")
    cache.add_argument("--hash", help="cache hash under $CACHE_DIR instead of the current one")
    args = parse_args(parser, argv)

    if args.hash is None:
        from bbTT.configs.io_config import DataConfig  # imports torch, only needed for the default

        args.hash = DataConfig().content_hash()
    cache_dir = pathlib.Path(os.environ["CACHE_DIR"]) / args.hash
    files = find_sizes_files(cache_dir, kind="pid" if args.pid else "dataset", eras=args.eras)
    print(f"cache: {cache_dir}")

    table = sum_sizes(files)

    formatters = {column: "{:,}".format for column in table.columns}
    with pd.option_context("display.max_rows", None, "display.width", None):
        if table.index.get_level_values("category").nunique() == 1:  # process ids: no categories
            print(table.droplevel("category").to_string(formatters=formatters))
        else:
            print(table.to_string(formatters=formatters))
            print("\nsubtotals:")
            print(table.groupby(level="category").sum().sort_values("total", ascending=False).to_string(formatters=formatters))
    return 0
