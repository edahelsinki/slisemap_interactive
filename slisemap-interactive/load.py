"""
    Load Slisemap objects and convert them into dataframes.
"""

import gc
from os import PathLike
from typing import Optional, Sequence, Union
import warnings

import pandas as pd
import numpy as np
from slisemap import Slisemap


def slisemap_to_dataframe(
    path: Union[str, PathLike, Slisemap],
    variables: Optional[Sequence[str]] = None,
    targets: Union[str, Sequence[str], None] = None,
    losses: bool = False,
    clusters: int = 9,
) -> pd.DataFrame:
    """Convert a `Slisemap` object to a `pandas.DataFrame`.

    Args:
        path: Slisemap object or path to a saved slisemap object.
        variables: List of variables names (columns in X). Defaults to None.
        targets: List of target names (columns in Y). Defaults to None.
        losses: Return the loss matrix. Default to False.
        clusters: Return cluster indices (if greater than one). Defaults to 9.

    Returns:
        A dataframe containing the X, Y, Z, B, Local loss, (L, Cluster) matrices from the Slisemap object.
    """
    # TODO slisemap column names

    if isinstance(path, Slisemap):
        sm = path
    else:
        sm = Slisemap.load(path, "cpu")

    dfs = []

    X = sm.get_X(intercept=False)
    if variables:
        assert len(variables) == X.shape[1]
        Xn = [f"X_{i}" for i in variables]
    else:
        Xn = [f"X_{i+1}" for i in range(X.shape[1])]
    dfs.append(pd.DataFrame.from_records(X, columns=Xn))
    del X

    Y = sm.get_Y()
    if targets is None and Y.shape[1] == 1:
        Yn = ["Y"]
    elif targets is None:
        Yn = [f"Y_{i+1}" for i in range(Y.shape[1])]
    else:
        if not isinstance(targets, str):
            targets = [targets]
        Yn = [f"Y_{i}" for i in targets]
    dfs.append(pd.DataFrame.from_records(Y, columns=Yn))
    del Y

    Z = sm.get_Z(rotate=True)
    Zn = [f"Z_{i+1}" for i in range(Z.shape[1])]
    dfs.append(pd.DataFrame.from_records(Z, columns=Zn))
    del Z

    B = sm.get_B()
    if variables:
        if sm.intercept:
            variables = variables + ["B_Intercept"]
        if B.shape[1] == len(variables):
            Bn = [f"B_{i}" for i in variables]
        elif B.shape[1] % len(variables) == 0 and B.shape[1] % len(targets) == 0:
            variables = [f"{t}:{v}" for t in targets for v in variables]
            Bn = [f"B_{i}" for i in variables[: B.shape[1]]]
        else:
            Bn = [f"B_{i+1}" for i in range(X.shape[1])]
    else:
        if sm.intercept:
            Bn = [f"B_{i+1}" for i in range(B.shape[1] - 1)] + ["B_Intercept"]
        else:
            Bn = [f"B_{i+1}" for i in range(B.shape[1])]
    dfs.append(pd.DataFrame.from_records(B, columns=Bn))
    del B

    L = sm.get_L()
    dfs.append(pd.DataFrame({"Local loss": L.diagonal()}))
    if losses:
        Ln = [f"L_{i+1}" for i in range(L.shape[1])]
        dfs.append(pd.DataFrame.from_records(L, columns=Ln))
    del L

    if clusters > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            clusters = {
                f"Clusters {i}": pd.Series(
                    sm.get_model_clusters(i)[0], dtype="category"
                )
                for i in range(2, clusters + 1)
            }
            dfs.append(pd.DataFrame(clusters))

    # Then we create a dataframe to return
    del sm
    gc.collect(1)
    df = pd.concat(dfs, axis=1)
    del dfs
    gc.collect(1)
    return df
