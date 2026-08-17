from dataclasses import dataclass

@dataclass(frozen=True)
class LevelSpec():
    num: int
    name: str
    color: str
    doc: str = ""


RESET =     "\x1b[0m"

# foreground colors
_FG_RED =        "\x1b[91m"
_FG_GREEN =      "\x1b[92m"
_FG_YELLOW =     "\x1b[93m"
_FG_BLUE =       "\x1b[94m"
_FG_MAGENTA =    "\x1b[95m"
_FG_CYAN =       "\x1b[96m"
_FG_WHITE =         "\x1b[97m"
_FG_DARK_YELLOW =    "\x1b[33m"  # dark yellow for debugging

_BG_MAGENTA =    "\x1b[45m"  # bright magenta
_BG_RED =        "\x1b[41m"  # red
_BG_YELLOW =     "\x1b[43m"  # yellow
_BG_GREEN =      "\x1b[42m"  # green
_BG_CYAN =       "\x1b[46m"  # cyan
_BG_ORANGE =     "\x1b[48;5;208m"  # bright orange
_BG_YELLOW_FG_RED = "\x1b[31;43m"  # bright yellow background with red foreground


# log level colors, needs to be the same name as the level name registered in get_logger
STANDARD_LEVEL_COLORS = {
    "CRITICAL" : _BG_MAGENTA,
    "ERROR" : _BG_RED,
    "WARNING" : _BG_ORANGE,
    "INFO" : _FG_YELLOW,
    "DEBUG" : _FG_DARK_YELLOW,
}

CUSTOM_LEVELS = [
    LevelSpec(num=21, name="I_INFO", color=_BG_YELLOW_FG_RED, doc="Important info for the user"),
    LevelSpec(num=25, name="TRAINING", color=_FG_WHITE, doc="Training process updates")
]

COLORS = {**STANDARD_LEVEL_COLORS, **{spec.name: spec.color for spec in CUSTOM_LEVELS}}
