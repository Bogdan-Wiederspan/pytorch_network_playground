import typing


def choice_check(selected, choices):
    # helper to verify that a selected value is part of a set of valid choices, can be used for runtime checks of config values
    choices = typing.get_args(choices)
    if selected not in choices:
        raise ValueError(f"Selected ({selected}) is not part of valid choices {choices}")
