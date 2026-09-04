import warnings

import numpy as np

from bbTT.utils.utils import EMPTY_FLOAT


def compute_fn(df, column_name, compute_fn):
    """Return a column, computing and caching it only if absent.

    Args:
        df (pandas.DataFrame): DataFrame to read from and write into.
        column_name (str): Name of the column to fetch or create.
        compute_fn (callable): No-argument function that computes the column if missing.

    Returns (pandas.Series):
        The requested column.
    """
    if column_name not in df.columns:
        df[column_name] = compute_fn()
        num_nan = df[column_name].isna().sum()
        if num_nan > 0:
            warnings.warn(f"Warning: {num_nan} NaN values in {column_name}, replacing with 0.")
            df[column_name] = df[column_name].fillna(EMPTY_FLOAT)
    return df[column_name]


def transversal_impuls(df, column_name):
    # pt = sqrt(px^2 + py^2)
    def fn():
        return np.sqrt(df[f"{column_name}_px"] ** 2 + df[f"{column_name}_py"] ** 2)

    col = f"{column_name}_pt"
    return compute_fn(df, col, fn)


def pseudorapidity(df, column_name):
    # eta = 0.5 * ln((p+pz)/(p-pz))
    def fn():
        p = momentum(df, column_name)
        return 0.5 * np.log((p + df[f"{column_name}_pz"]) / (p - df[f"{column_name}_pz"]))

    col = f"{column_name}_eta"
    return compute_fn(df, col, fn)


def momentum(df, column_name):
    # p = sqrt(px^2 + py^2 + pz^2)
    def fn():
        return np.sqrt(df[f"{column_name}_px"] ** 2 + df[f"{column_name}_py"] ** 2 + df[f"{column_name}_pz"] ** 2)

    col = f"{column_name}_p"
    return compute_fn(df, col, fn)


def azimuthal_angle(df, column_name):
    # tan (theta) = py/px
    def fn():
        return np.arctan2(df[f"{column_name}_py"], df[f"{column_name}_px"])  # compute phi in range [-pi, pi]

    col = f"{column_name}_phi"
    return compute_fn(df, col, fn)


def invariant_mass(df, column_name):
    # m^2 = E^2 - p^2
    def fn():
        p = momentum(df, column_name)
        return np.sqrt(df[f"{column_name}_e"] ** 2 - p**2)

    col = f"{column_name}_mass"
    return compute_fn(df, col, fn)


def transverse_mass(df, column_name):
    # mT^2 = E^2 - pT^2
    def fn():
        pt = transversal_impuls(df, column_name)
        return np.sqrt(df[f"{column_name}_e"] ** 2 - pt**2)

    col = f"{column_name}_mt"
    return compute_fn(df, col, fn)


def compute_derived(df, column_names):
    for col in column_names:
        transversal_impuls(df, col)
        pseudorapidity(df, col)
        momentum(df, col)
        azimuthal_angle(df, col)
        invariant_mass(df, col)
        transverse_mass(df, col)
