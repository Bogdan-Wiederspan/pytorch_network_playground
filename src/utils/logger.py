import logging
import sys
import os
from pathlib import Path

RED_LEVEL = 21
GREEN_LEVEL = 22
YELLOW_LEVEL = 23
BLUE_LEVEL = 24
MAGENTA_LEVEL = 25
CYAN_LEVEL = 26

_color_levels_registered = False

# helper function to add log methods for custom levels
def _make_log_method(level_num):
    def log_method(self, msg, *args, **kwargs):
        if self.isEnabledFor(level_num):
            self._log(level_num, msg, args, **kwargs)
    return log_method


def _register_color_levels():
    global _color_levels_registered
    if _color_levels_registered:
        return

    # color levels mapping that are used in ColoredFormatter, add them to logging and create methods for them
    color_levels = {
        RED_LEVEL: "RED",
        GREEN_LEVEL: "GREEN",
        YELLOW_LEVEL: "YELLOW",
        BLUE_LEVEL: "BLUE",
        MAGENTA_LEVEL: "MAGENTA",
        CYAN_LEVEL: "CYAN",
    }
    # register the levels and add methods to the Logger class
    for level_num, level_name in color_levels.items():
        logging.addLevelName(level_num, level_name)
        setattr(logging.Logger, level_name.lower(), _make_log_method(level_num))

    _color_levels_registered = True


def get_logger(name=None):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = ColoredFormatter(fmt="[%(asctime)s] %(levelname)s %(funcName)s: %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        _register_color_levels()

    return logger



class ColoredFormatter(logging.Formatter):
    RESET =     "\x1b[0m"

    # foreground colors
    RED =       "\x1b[91m"
    GREEN =     "\x1b[92m"
    YELLOW =    "\x1b[93m"
    BLUE =      "\x1b[94m"
    MAGENTA =   "\x1b[95m"
    CYAN =      "\x1b[96m"

    # background colors
    CRITICAL =  "\x1b[45m"  # bright magenta
    ERROR =     "\x1b[41m"  # red
    WARNING =   "\x1b[43m"  # yellow
    INFO =      "\x1b[42m"  # green
    DEBUG =     "\x1b[46m"  # cyan

    def __init__(self, fmt=None, datefmt=None, style="%", use_color=True):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self.use_color = use_color and sys.stdout.isatty()

    def format(self, record):
        # get the original log message
        message = super().format(record)
        if self.use_color:
            color = getattr(self, record.levelname)
            # color = self.COLORS.get(record.levelname, "")
            return f"{color}{message}{self.RESET}"
        return message

class TensorboardLogger():
    def __init__(self, name=None, path=None):
        self.hash = name.stem
        self.path = self.logger_path() if path is None else self.log_dir / path
        self.writer = self.create_tensorboard_writer(log_dir=self.path)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_loss(self, values, step):
        self.writer.add_scalars("Loss", values, global_step=step)

    def log_scalar(self, values, step, name):
        self.writer.add_scalars(name, values, global_step=step)

    def log_lr(self, value, step):
        self.writer.add_scalar("Learning_Rate", value, step)

    def log_precision(self, values, step, mode="train"):
        acc = {k : v["precision"] for k, v in values.items()}
        self.writer.add_scalars(f"{mode} Precision", acc, step)

    def log_f1(self, values, step, mode="train"):
        acc = {k : v["f1"] for k, v in values.items()}
        self.writer.add_scalars(f"{mode} F1-Score", acc, step)

    def log_sensitivity(self, values, step, mode="train"):
        acc = {k: v["sensitivity"] for k, v in values.items()}
        self.writer.add_scalars(f"{mode} Recall", acc, step)

    def log_histogram(self, tag, values, step):
        self.writer.add_histogram(tag, values, step)

    def log_figure(self, tag, figure, step):
        self.writer.add_figure(tag, figure, step)

    @property
    def log_dir(self):
        return Path(os.environ["TENSORBOARD_DIR"])

    def logger_path(self):
        from time import localtime, strftime
        t = strftime("%Y_%m_%d-%H_%M_%S", localtime())
        new_stem = f"{t}-{self.hash}"
        logger_path = self.log_dir / new_stem
        return logger_path

    def create_tensorboard_writer(self, config=None, log_dir=None):
        # TODO: THINK about location of writer? Should it be inside the hashed dir?
        # from data.cache import DataCacher
        from torch.utils.tensorboard import SummaryWriter
        # if config is not None:
        #     cacher = DataCacher(config)
        #     if log_dir is None:
        #         log_dir = cacher.cache_dir()
        return SummaryWriter(log_dir=log_dir)
