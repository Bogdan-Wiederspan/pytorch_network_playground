
"""
Sum the per-key counts of the ``<era>_{dataset,pid}_sizes.json`` files of a data cache into one table.

The cache lives in ``$CACHE_DIR/<hash>``, the hash defaults to the one of the current ``DataConfig``.

Usage: ``bbtt-sum-sizes --current`` or ``bbtt-sum-sizes [--pid] [--before-cut] [--eras 22pre 22post] [--hash <hash>]``
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



def load_counts(path: pathlib.Path, before_cut: bool = False) -> dict[str, int]:
    """
    Load one sizes file as ``{key: count}``.

    ``<era>_pid_sizes.json`` holds plain counts. ``<era>_dataset_sizes.json`` holds per dataset
    a list of ``[file_path, entries_in_tree, entries_after_cut]`` per root file (see
    ``root_reading.load_root_and_convert_to_numpy``), which is summed over the files here.

    Args:
        path (pathlib.Path): Sizes file.
        before_cut (bool): For dataset sizes, sum the tree entries instead of the entries after the cut.

    Returns:
        dict[str, int]: Count per key.
    """
    index = 1 if before_cut else 2
    return {
        key: sum(entry[index] for entry in value) if isinstance(value, list) else value
        for key, value in json.loads(path.read_text()).items()
    }


def sum_sizes(files: list[pathlib.Path], before_cut: bool = False) -> pd.DataFrame:
    """
    Build the summed table: one column per era, a total, grouped by category.

    The category is the first token of a dataset name (``tt_sl_powheg`` -> ``tt``); plain
    integer keys (process ids) all fall into one unnamed category.

    Args:
        files (list[pathlib.Path]): Sizes files named ``<era>_<kind>.json``, one per era.
        before_cut (bool): Passed to :func:`load_counts`.


    Returns:
        pd.DataFrame: Index (category, key), one column per era plus ``total``; sorted by
        category subtotal, then by total, descending; process ids ascending by id instead.
        Missing keys count as 0.
    """
    # column name = era = file name up to the first "_"

    table = pd.DataFrame({path.name.partition("_")[0]: load_counts(path, before_cut) for path in files}).fillna(0).astype(int)
    table["total"] = table.sum(axis=1)
    table["category"] = ["" if key.isdigit() else key.partition("_")[0] for key in table.index]

    if (table["category"] == "").all():  # process ids: numeric order of the id
        table = table.loc[sorted(table.index, key=int)]
    else:
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
    parser.add_argument("--before-cut", action="store_true", help="dataset sizes: count the tree entries instead of the events after the cut")

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

    table = sum_sizes(files, before_cut=args.before_cut)

    formatters = {column: "{:_}".format for column in table.columns}
    with pd.option_context("display.max_rows", None, "display.width", None):
        if table.index.get_level_values("category").nunique() == 1:  # process ids: no categories
            print(table.droplevel("category").to_string(formatters=formatters))
        else:
            print(table.to_string(formatters=formatters))
            print("\nsubtotals:")
            print(table.groupby(level="category").sum().sort_values("total", ascending=False).to_string(formatters=formatters))
    return 0
