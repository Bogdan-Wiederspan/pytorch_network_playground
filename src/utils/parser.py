import argparse

class ParserBuilder():
    def __init__(self, *args):
        self.parser = argparse.ArgumentParser()
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

    def build(self, args):
        commands = [f"add_{arg}" for arg in args]
        for command in commands:
            getattr(self, command)()
