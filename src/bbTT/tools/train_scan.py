"""
Run a series of trainings that differ in config overrides: a grid, explicit points or a scan file, one after another.

Usage: ``bbtt-train-scan -sn lrscan --scan optimizer_config.lr=1e-3,3e-3 --scan model_building_config.dropout=0.1,0.3 --dry-run``
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import re
import shlex

from bbTT.tools.cli import parse_args

Point = dict[str, str]  # override path -> value, both as typed on the command line


def parse_scan(spec: str) -> tuple[str, list[str]]:
    """
    Split one ``--scan PATH=V1,V2,...`` argument.

    Args:
        spec (str): E.g. ``optimizer_config.lr=1e-3,3e-3``.

    Returns:
        tuple[str, list[str]]: Path and its values.

    Raises:
        ValueError: If no ``=`` or no value is given.
    """
    path, separator, values = spec.partition("=")
    values = [value.strip() for value in values.split(",") if value.strip()]
    if not separator or not values:
        raise ValueError(f"--scan '{spec}': expected PATH=VALUE[,VALUE...]")
    return path.strip(), values


def parse_point(specs: list[str]) -> Point:
    """
    Args:
        specs (list[str]): ``PATH=VALUE`` entries of one ``--point``.

    Returns:
        Point: Path -> value.

    Raises:
        ValueError: If an entry has no ``=``.
    """
    point = {}
    for spec in specs:
        path, separator, value = spec.partition("=")
        if not separator or not value.strip():
            raise ValueError(f"--point entry '{spec}': expected PATH=VALUE")
        point[path.strip()] = value.strip()
    return point


def grid_points(grid: dict[str, list[str]]) -> list[Point]:
    """
    Cartesian product of the scanned values, in the order the paths were given.

    Args:
        grid (dict[str, list[str]]): Path -> values.

    Returns:
        list[Point]: One point per combination; empty for an empty grid.
    """
    if not grid:
        return []
    return [dict(zip(grid, combination)) for combination in itertools.product(*grid.values())]


def load_scan_file(path: pathlib.Path) -> tuple[list[Point], list[str]]:
    """
    Read a scan file: ``{"grid": {PATH: [values]}, "points": [{PATH: value}], "overrides": {PATH: value}}``,
    every key optional, values may be numbers (converted to the string the command line would carry).

    Args:
        path (pathlib.Path): JSON file.

    Returns:
        tuple[list[Point], list[str]]: Points (grid product plus explicit points) and fixed overrides as ``PATH=VALUE``.
    """
    content = json.loads(path.read_text())
    grid = {key: [str(value) for value in values] for key, values in content.get("grid", {}).items()}
    points = grid_points(grid) + [{key: str(value) for key, value in point.items()} for point in content.get("points", [])]
    overrides = [f"{key}={value}" for key, value in content.get("overrides", {}).items()]
    return points, overrides


def point_tag(point: Point) -> str:
    """
    Readable run suffix from the scanned fields only: last path segment plus value, e.g. ``lr1e-3-dropout0.1``.

    Args:
        point (Point): Path -> value.

    Returns:
        str: Tag, characters outside ``[A-Za-z0-9.+-]`` replaced by ``_``.
    """
    parts = [f"{path.rsplit('.', 1)[-1]}{value}" for path, value in point.items()]
    return re.sub(r"[^A-Za-z0-9.+-]", "_", "-".join(parts))


def train_command(point: Point, tensorboard_name: str, save_model_name: str, fixed: list[str], extra: list[str]) -> list[str]:
    """
    The equivalent ``bbtt-train`` call of one point, e.g. to hand it to a batch system.

    Args:
        point (Point): Scanned overrides of the point.
        tensorboard_name (str): Resolved tensorboard name of the point.
        save_model_name (str): Resolved checkpoint name of the point.
        fixed (list[str]): Overrides shared by all points, ``PATH=VALUE``.
        extra (list[str]): Further bbtt-train arguments passed through (cache flags, checkpoint suffix).

    Returns:
        list[str]: Command as argv.
    """
    overrides = fixed + [f"{path}={value}" for path, value in point.items()]
    return ["bbtt-train", "-tn", tensorboard_name, "-sn", save_model_name, *extra, "-o", *overrides]


def main(argv: list[str] | None = None) -> int:
    """
    Entry point of ``bbtt-train-scan``.

    Args:
        argv (list[str] | None): Command line arguments without the program name.

    Returns:
        int: Exit code, 0 on success; the first failing point aborts the scan with its exception.
    """
    parser = argparse.ArgumentParser(prog="bbtt-train-scan", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save_model_name", "-sn", required=True, help="base name, every point gets '<name>-<tag>' with the tag built from its scanned values")
    parser.add_argument("--tensorboard_name", "-tn", default=None, help="base tensorboard name, same scheme (default: the checkpoint base name)")
    parser.add_argument("--scan", action="append", default=[], metavar="PATH=V1,V2,...", help="values of one path, repeat for more; the points are the cartesian product")
    parser.add_argument("--point", action="append", nargs="+", default=[], metavar="PATH=VALUE", help="one explicit point (several PATH=VALUE), repeat for more; added to the grid points")
    parser.add_argument("--file", type=pathlib.Path, default=None, help="scan file with 'grid', 'points' and 'overrides' (json)")
    parser.add_argument("--override", "-o", dest="overrides", nargs="*", default=[], metavar="PATH=VALUE", help="overrides shared by all points, applied first")
    parser.add_argument("--checkpoint_suffix", "-cs", default=None, help="passed through to every training")
    parser.add_argument("--rebuild_cache", "-rc", action="store_true", help="passed through to every training")
    parser.add_argument("--no_save_cache", "-ns", action="store_true", help="passed through to every training")
    parser.add_argument("--dry-run", action="store_true", help="list the points and their run names, train nothing")
    parser.add_argument("--print-commands", action="store_true", help="print one bbtt-train call per point (e.g. for a batch system), train nothing")
    args = parse_args(parser, argv)

    grid = dict(parse_scan(spec) for spec in args.scan)
    points = grid_points(grid) + [parse_point(specs) for specs in args.point]
    fixed = list(args.overrides)
    if args.file is not None:
        file_points, file_overrides = load_scan_file(args.file)
        points += file_points
        fixed = file_overrides + fixed  # command line overrides win over the file
    if not points:
        parser.error("no points: give --scan, --point or --file")

    tensorboard_base = args.tensorboard_name or args.save_model_name
    extra = (["-cs", args.checkpoint_suffix] if args.checkpoint_suffix else []) + (["-rc"] if args.rebuild_cache else []) + (["-ns"] if args.no_save_cache else [])
    runs = [(point, f"{tensorboard_base}-{point_tag(point)}", f"{args.save_model_name}-{point_tag(point)}") for point in points]

    if args.print_commands:
        for point, tensorboard_name, save_model_name in runs:
            print(shlex.join(train_command(point, tensorboard_name, save_model_name, fixed, extra)))
        return 0

    width = max(len(save_model_name) for _, _, save_model_name in runs)
    print(f"{len(runs)} points, shared overrides: {' '.join(fixed) or 'none'}")
    for index, (point, _, save_model_name) in enumerate(runs, start=1):
        print(f"  {index:>3}  {save_model_name:<{width}}  {' '.join(f'{path}={value}' for path, value in point.items())}")
    if args.dry_run:
        return 0

    from bbTT.train.train import train  # local import: torch, only needed to run

    for index, (point, tensorboard_name, save_model_name) in enumerate(runs, start=1):
        print(f"\n=== point {index}/{len(runs)}: {save_model_name} ===")
        # an exception aborts the whole scan, the remaining points are listed above for a restart
        train(
            tensorboard_name=tensorboard_name,
            save_model_name=save_model_name,
            checkpoint_suffix=args.checkpoint_suffix,
            rebuild_cache=args.rebuild_cache,
            save_cache=not args.no_save_cache,
            overrides=fixed + [f"{path}={value}" for path, value in point.items()],
        )
    return 0
