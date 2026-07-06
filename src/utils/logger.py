from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_custom_levels_registered = False

# helper function to add log methods for custom levels
def _make_log_method(level_num):
    def log_method(self, msg, *args, stacklevel=2, **kwargs):
        if self.isEnabledFor(level_num):
            # if stacklevel is not provided, default to 2 to point to the caller of the log method
            # otherwise it would point to the log method itself
            if "stacklevel" not in kwargs:
                kwargs["stacklevel"] = stacklevel
            self._log(level_num, msg, args, **kwargs)
    return log_method

def _register_custom_log_levels():
    global _custom_levels_registered
    if _custom_levels_registered:
        return

    # color levels mapping that are used in ColoredFormatter, add them to logging and create methods for them
    custom_levels = {
        21: "I_INFO",
        25: "TRAINING",
    }
    # register the levels and add methods to the Logger class
    for level_num, level_name in custom_levels.items():
        logging.addLevelName(level_num, level_name)
        # create logging method with the same name as level name
        setattr(logging.Logger, level_name.lower(), _make_log_method(level_num))

    _custom_levels_registered = True


def get_logger(name:str ="root", file_path=None) -> logging.Logger:
    """
    Helper function to create a logger with a specific name and configure it!
    A good tutorial can be found here: https://realpython.com/python-logging/
    Args:
        name (str, optional): Name of the logger. Defaults to root.

    Returns:
        logging.Logger: A logger instance with the specified name and configuration.
    """
    # log level of the console handler, file handler logs everything
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    file_log_level = os.environ.get("FILE_LOG_LEVEL", "DEBUG").upper()

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if not logger.hasHandlers():
        # stream handler to print logs to console per stderr
        console_handler = logging.StreamHandler(stream=sys.stderr)
        console_handler.setLevel(log_level)
        # full list of predefined attributes https://docs.python.org/3/library/logging.html#logrecord-attributes
        # levelname = name of log level
        # funcName = name of function that called the logger
        # asctime = time of log message
        # lineno = line number in the code where the log message was called
        # message = the log message

        # HINT: f strings are eagerly evaluated, THUS ALWAYS EVALUATED even if log level is not enabled
        # using % style formatting to overcome this
        formatter_string = "[%(asctime)s] %(levelname)s L:%(lineno)s-%(funcName)s: %(message)s"
        # set up formatter that matches log name with a color defined in ColoredFormatter
        formatter = ColoredFormatter(fmt=formatter_string, datefmt="%H:%M:%S")

        # TODO add file handler to log to a file, maybe in the same directory as the tensorboard logs? Or in a separate logs directory?
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if file_path is not None:
            file_handler = logging.FileHandler(file_path, mode="a", encoding="utf-8")
            file_formatter = logging.Formatter(fmt=formatter_string, datefmt="%Y-%m-%d %H:%M:%S")
            file_handler.setFormatter(file_formatter) # use different formatter to remove coloring
            file_handler.setLevel(file_log_level)
            logger.addHandler(file_handler)

        _register_custom_log_levels()

    return logger



class ColoredFormatter(logging.Formatter):
    RESET =     "\x1b[0m"

    # foreground colors
    FG_RED =        "\x1b[91m"
    FG_GREEN =      "\x1b[92m"
    FG_YELLOW =     "\x1b[93m"
    FG_BLUE =       "\x1b[94m"
    FG_MAGENTA =    "\x1b[95m"
    FG_CYAN =       "\x1b[96m"
    FG_WHITE =         "\x1b[97m"
    FG_DARK_YELLOW =    "\x1b[33m"  # dark yellow for debugging

    BG_MAGENTA =    "\x1b[45m"  # bright magenta
    BG_RED =        "\x1b[41m"  # red
    BG_YELLOW =     "\x1b[43m"  # yellow
    BG_GREEN =      "\x1b[42m"  # green
    BG_CYAN =       "\x1b[46m"  # cyan
    BG_ORANGE =     "\x1b[48;5;208m"  # bright orange
    BG_YELLOW_FG_RED = "\x1b[31;43m"  # bright yellow background with red foreground


    # log level colors, needs to be the same name as the level name registered in get_logger
    CRITICAL =  BG_MAGENTA
    ERROR =     BG_RED
    WARNING =   BG_ORANGE           # WARNINGS TO indicate potential problems that are handled for you
    INFO =      FG_YELLOW           # WHAT DOES THE PROGRAM DO CURRENTLY
    I_INFO =    BG_YELLOW_FG_RED    # IMPORTANT INFORMATION FOR THE USER
    TRAINING =  FG_WHITE               # TRAININGS PROCESS
    DEBUG =     FG_DARK_YELLOW             # DETAILED DEBUGGING INFORMATION FOR DEVELOPERS
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
        self.hash = name
        self.path = self.logger_path() if path is None else self.log_dir / path
        self.writer = self.create_tensorboard_writer(log_dir=self.path)

    def log_loss(self, values, step):
        self.writer.add_scalars("Loss", values, global_step=step)

    def log_scalar(self, values, step, name):
        self.writer.add_scalars(name, values, global_step=step)

    def log_lr(self, optimizer, step):
        value = optimizer.param_groups[0]["lr"]
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
