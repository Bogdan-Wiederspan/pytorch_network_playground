from __future__ import annotations

import logging
import os
import sys

from bbTT.monitoring.logger.formatter import ColoredFormatter
from bbTT.monitoring.logger.log_level import CUSTOM_LEVELS

_custom_levels_registered = False

class SameLineStreamHandler(logging.StreamHandler):
    """
    A StreamHandler that understands "same-line" records (tagged via
    extra={"same_line": True}, see _make_same_line_log_method).

    - same_line record, first one in a row: written normally, no trailing
    newline, cursor stays at end of line.
    - same_line record, follows another same_line record: prefixed with
    "\r" so it overwrites the previous same-line text in place.
    - any other record: if a same-line record was pending, a newline is
    written first to close out that line cleanly, then this record is
    written normally (respecting its own terminator).

    This state lives on the handler, not in the log methods, so it works
    correctly no matter which logger method (custom or stdlib) writes next.
    """

    def __init__(self, stream=None):
        super().__init__(stream)
        self._progress_active = False

    def emit(self, record):
        same_line = getattr(record, "same_line", False)
        try:
            message = self.format(record)
            stream = self.stream
            if same_line:
                prefix = "\r" if self._progress_active else ""
                stream.write(prefix + message)
                self._progress_active = True
            else:
                if self._progress_active:
                    stream.write("\n")
                    self._progress_active = False
                stream.write(message + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

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

def _make_same_line_log_method(level_num: int):
    """
    Logs on the current line (without newline) - e.g. for progress updates.

    Just tags the record as same_line=True via `extra`; SameLineStreamHandler
    decides how to render it (overwrite vs. start fresh vs. break to a new
    line for whatever comes after). No handler state is touched here.

    Args:
        level_num (int): Log Level of the method.
    """
    def log_method(self, msg, *args, stacklevel=2, **kwargs):
        if self.isEnabledFor(level_num):
            if "stacklevel" not in kwargs:
                kwargs["stacklevel"] = stacklevel
            extra = kwargs.setdefault("extra", {})
            extra["same_line"] = True
            self._log(level_num, msg, args, **kwargs)
    return log_method

def _register_custom_log_levels():
    global _custom_levels_registered
    if _custom_levels_registered:
        return

    # register the levels and add methods to the Logger class
    for spec in CUSTOM_LEVELS:
        logging.addLevelName(spec.num, spec.name)
        method_name = spec.name.lower()
        # create logging method with the same name as level name, but lower
        setattr(logging.Logger, method_name, _make_log_method(spec.num))
        setattr(logging.Logger, f"{method_name}_progress", _make_same_line_log_method(spec.num))

    # each normal logger should
    setattr(logging.Logger, "debug_progress", _make_same_line_log_method(logging.DEBUG))
    setattr(logging.Logger, "info_progress", _make_same_line_log_method(logging.INFO))

    _custom_levels_registered = True


def get_logger(name:str ="root", file_path: str | None=None) -> logging.Logger:
    """
    Factory to create / get a named logger instance with optional file write out.

    Level of the logger is taken from environment:
        Console handler level: env LOG_LEVEL (default: INFO)
        File handler level: env FILE_LOG_LEVEL (default: DEBUG)

    A good tutorial can be found here: https://realpython.com/python-logging/

    Args:
        name (str, optional): Name of the logger. Defaults to root.
        file_path (str, optional): If given, also log to this file, appending.

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
        console_handler = SameLineStreamHandler(stream=sys.stderr)
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
