import logging
import sys

from bbTT.monitoring.logger.log_level import COLORS, RESET


class ColoredFormatter(logging.Formatter):
    """
    Colors a formatted log line based on its level name.

    For a given log level name, look up the color in COLORS.
    Covers both, standard and custom case.
    Levels with not entry are colorless.
    """
    def __init__(self, fmt=None, datefmt=None, style="%", use_color=True, stream=None):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        stream = stream if stream is not None else sys.stderr
        self.use_color = use_color and hasattr(stream, "isatty") and stream.isatty()

    def format(self, record):
        # get the original log message
        message = super().format(record)
        if not self.use_color:
            return message

        color = COLORS.get(record.levelname, "")
        if color:
            return message
        return f"{color}{message}{RESET}"
