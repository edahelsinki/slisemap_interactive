"""
    Load Slisemap objects and convert them into dataframes.
"""

import gc
from os import PathLike
from typing import Union
import warnings

import pandas as pd
import numpy as np
from slisemap import Slisemap
from sklearn.cluster import KMeans


def subsample(Z: np.ndarray, n: int, cluster: bool = True) -> np.ndarray:
    """Get indices for subsampling.
    Optionally uses k-means clustering to ensure inclusion of rarer data items.

    Args:
        Z: Embedding matrix.
        n: Size of the suvset.
        cluster: Use k-means. Defaults to True.

    Returns:
        A list of indices.
    """
    if n >= Z.shape[0]:
        return np.arange(n)
    elif not cluster:
        return np.random.choice(Z.shape[0], n, replace=False)
    else:
        selected = np.random.choice(Z.shape[0], n, replace=False)
        nc = min(n // 10, 100)
        lc = KMeans(nc, n_init=5).fit(Z).labels_
        lc[selected[nc:]] = -1
        for c in range(nc):
            wc = np.nonzero(lc == c)[0]
            if wc.size > 0:
                selected[c] = np.random.choice(wc, 1)
        return selected


def slisemap_to_dataframe(
    path: Union[str, PathLike, Slisemap],
    losses: bool = True,
    clusters: int = 9,
    max_n: int = -1,
) -> pd.DataFrame:
    """Convert a `Slisemap` object to a `pandas.DataFrame`.

    Args:
        path: Slisemap object or path to a saved slisemap object.
        losses: Return the loss matrix. Default to True.
        clusters: Return cluster indices (if greater than one). Defaults to 9.
        max_n: maximum number of data items in the dataframe (subsampling is recommended if `n > 5000` and `losses=True`). Defaults to -1.

    Returns:
        A dataframe containing data from the Slisemap object (columns: "X_*", "Y_*", "Z_*", "B_*", "Local loss", ("L_*", "Clusters *")).
    """
    if isinstance(path, Slisemap):
        sm = path
    else:
        sm = Slisemap.load(path, "cpu")

    if max_n > 0 and sm.n > max_n:
        ss = subsample(sm.get_Z(), max_n)
    else:
        ss = ...

    def preface_names(names, preface):
        return [n if n[:2] == preface else preface + n for n in map(str, names)]

    variables = sm.metadata.get_variables(intercept=False)
    variables = preface_names(variables, "X_")
    targets = sm.metadata.get_targets()
    if len(targets) > 1 or targets[0] != "Y":
        targets = preface_names(targets, "Y_")
    coefficients = sm.metadata.get_coefficients()
    coefficients = preface_names(coefficients, "B_")
    dimensions = sm.metadata.get_dimensions()
    dimensions = preface_names(dimensions, "Z_")
    rows = sm.metadata.get_rows(fallback=False)
    if ss is not ...:
        rows = ss if rows is None else np.asarray(rows)[ss]
    elif rows is None:
        rows = range(sm.n)

    dfs = [
        pd.DataFrame.from_records(sm.metadata.unscale_X()[ss, :], columns=variables),
        pd.DataFrame.from_records(sm.metadata.unscale_Y()[ss, :], columns=targets),
        pd.DataFrame.from_records(sm.get_Z(rotate=True)[ss, :], columns=dimensions),
        pd.DataFrame.from_records(sm.get_B()[ss, :], columns=coefficients),
    ]
    gc.collect(1)

    L = sm.get_L(X=sm._X[ss, :], Y=sm._Y[ss, :])[ss, :]
    dfs.append(pd.DataFrame({"Local loss": L.diagonal()}))
    if losses:
        Ln = [f"L_{i}" for i in rows]
        dfs.append(pd.DataFrame.from_records(L, columns=Ln))
    del L
    gc.collect(1)

    if clusters > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            cl = lambda i: sm.get_model_clusters(i)[0][ss]
            clusters = {
                f"Clusters {i}": pd.Series(cl(i), dtype="category")
                for i in range(2, clusters + 1)
            }
            dfs.append(pd.DataFrame(clusters))

    # Then we create a dataframe to return
    del sm
    gc.collect(1)
    df = pd.concat(dfs, axis=1, copy=False)
    df.index = rows
    del dfs
    gc.collect(1)
    return df
