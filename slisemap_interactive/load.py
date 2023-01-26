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
    losses: bool = True,
    clusters: int = 9,
) -> pd.DataFrame:
    """Convert a `Slisemap` object to a `pandas.DataFrame`.

    Args:
        path: Slisemap object or path to a saved slisemap object.
        losses: Return the loss matrix. Default to True.
        clusters: Return cluster indices (if greater than one). Defaults to 9.

    Returns:
        A dataframe containing data from the Slisemap object (columns: "X_*", "Y_*", "Z_*", "B_*", "Local loss", ("L_*", "Clusters *")).
    """
    if isinstance(path, Slisemap):
        sm = path
    else:
        sm = Slisemap.load(path, "cpu")

    variables = sm.metadata.get_variables(intercept=False)
    variables = [v if v[:2] == "X_" else "X_" + v for v in variables]
    targets = sm.metadata.get_targets()
    if len(targets) > 1 or targets[0] != "Y":
        targets = [t if t[:2] == "Y_" else "Y_" + t for t in targets]
    coefficients = sm.metadata.get_coefficients()
    coefficients = [c if c[:2] == "B_" else "B_" + c for c in coefficients]
    dimensions = sm.metadata.get_dimensions()
    dimensions = [d if d[:2] == "Z_" else "Z_" + d for d in dimensions]
    rows = sm.metadata["rows"] if "rows" in sm.metadata else range(sm.n)

    dfs = []

    dfs.append(pd.DataFrame.from_records(sm.metadata.unscale_X(), columns=variables))
    dfs.append(pd.DataFrame.from_records(sm.metadata.unscale_Y(), columns=targets))
    dfs.append(pd.DataFrame.from_records(sm.get_Z(rotate=True), columns=dimensions))
    dfs.append(pd.DataFrame.from_records(sm.get_B(), columns=coefficients))
    gc.collect(1)

    L = sm.get_L()
    dfs.append(pd.DataFrame({"Local loss": L.diagonal()}))
    if losses:
        Ln = [f"L_{i}" for i in rows]
        dfs.append(pd.DataFrame.from_records(L, columns=Ln))
    del L

    if clusters > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            cl = sm.get_model_clusters
            clusters = {
                f"Clusters {i}": pd.Series(cl(i)[0], dtype="category")
                for i in range(2, clusters + 1)
            }
            dfs.append(pd.DataFrame(clusters))

    # Then we create a dataframe to return
    del sm
    gc.collect(1)
    df = pd.concat(dfs, axis=1)
    df.index = rows
    del dfs
    gc.collect(1)
    return df
