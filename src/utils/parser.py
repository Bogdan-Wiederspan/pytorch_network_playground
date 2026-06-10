import argparse
import pathlib


class ParserBuilder():
    def __init__(self, *args, description=""):
        self.parser = argparse.ArgumentParser(description=description)
        self.build(args)
        self.args = self.parser.parse_args()

    def add_tensorboard(self):
        self.parser.add_argument(
            "--tensorboard_name",
            "-tn",
            dest="tensorboard_name",
            action="store",
            default=None,
            help="Name of the tensorboard, if given, turns off generated name (default: None)"
            )

    def add_cache(self):
        self.parser.add_argument(
        "--ignore_cache",
        "-ic",
        action="store_true",
        default=False,
        help="Ignore cache when running the program (default: False)"
        )

        self.parser.add_argument(
        "--save-cache",
        "-s",
        dest="save_cache",
        action="store_true",
        default=False,
        help="Save cache (default: False)"
        )

    def add_load_checkpoint(self):

        self.parser.add_argument(
            "--src",
            "-s",
            dest="path",
            type=pathlib.Path,
            required=True,
            action="store",
            help="Path to model checkpoint to load, typically with .pt suffix"
        )

        def parse_folds(value):
            return [int(x) for x in value.split(",")]

        self.parser.add_argument(
            "--fold",
            "-f",
            dest="fold",
            required=True,
            default="0",
            type=parse_folds,
            help="Comma separated list of folds to pick the correct models and their respective test data."
        )

    def add_activation_fn(self):
        self.parser.add_argument(
            "--add_activation",
            required=False,
            help="If value is given, get activation function and add at end of network",
            default=None,
            choices=["sigmoid", "softmax", None]
        )


    def build(self, args):
        commands = [f"add_{arg}" for arg in args]
        for command in commands:
            getattr(self, command)()
