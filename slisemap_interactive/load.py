"""Load Slisemap objects and convert them into dataframes."""

import gc
import warnings
from os import PathLike
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

try:
    from slisemap import Slisemap
except ImportError:
    warnings.warn(
        "Could not import Slisemap, only limited functionality is available (no loading only plotting)",
        stacklevel=1,
    )

    class Slisemap:
        """Placeholder Slisemap class."""

        @classmethod
        def load(cls, *args: Any, **kwargs: Any) -> "Slisemap":
            """Trigger the loading from the real Slisemap."""
            from slisemap import Slisemap

            return Slisemap.load(*args, **kwargs)


try:
    from slisemap import Slipmap
except ImportError:
    warnings.warn(
        "Could not import Slipmap, only limited functionality is available (no loading only plotting)",
        stacklevel=1,
    )

    class Slipmap:
        """Placeholder Slipmap class."""

        @classmethod
        def load(cls, *args: Any, **kwargs: Any) -> "Slipmap":
            """Trigger the loading from the real Slipmap."""
            from slisemap import Slipmap

            return Slipmap.load(*args, **kwargs)


# Defaults for subsampling the Slisemap object
DEFAULT_MAX_N = 5000
DEFAULT_MAX_L = 250


def subsample(Z: np.ndarray, n: int, clusters: Optional[int] = None) -> np.ndarray:
    """Get indices for subsampling.

    Optionally uses k-means clustering to ensure inclusion of rarer data items.

    Args:
        Z: Embedding matrix.
        n: Size of the subset.
        clusters: Use k-means clustering to find a part of the subset. Defaults to `min(n//2, 100)`.

    Returns:
        An array of indices.
    """
    if n >= Z.shape[0]:
        return np.arange(Z.shape[0])
    if clusters is None:
        clusters = min(n // 2, 100)
    if clusters < 1:
        return np.random.choice(Z.shape[0], n, replace=False)
    else:
        indices = np.arange(Z.shape[0])
        dist = KMeans(clusters, n_init=5).fit_transform(Z)
        for i in range(clusters):
            item = np.argmin(dist[:, i])
            indices[[item, i]] = indices[[i, item]]
        for i in range(clusters, n):
            j = np.random.choice(Z.shape[0] - i, 1)[0] + i
            indices[[i, j]] = indices[[j, i]]
        return indices[:n]


def slisemap_to_dataframe(
    path: Union[str, PathLike, Slisemap],
    losses: Union[bool, int] = True,
    clusters: int = 9,
    max_n: int = -1,
    index: bool = True,
) -> pd.DataFrame:
    """Convert a `Slisemap` object to a `pandas.DataFrame`.

    Args:
        path: Slisemap object or path to a saved slisemap object.
        losses: Return the loss matrix. Can also be a number specifying the (approximate) maximum number of `L_*` columns. Default to True.
        clusters: Return cluster indices (if greater than one). Defaults to 9.
        max_n: maximum number of data items in the dataframe (subsampling is recommended if `n > 5000` and `losses=True`). Defaults to -1.
        index: Return row names as the index (True) or as an "item" column (False). Defaults to True.

    Returns:
        A dataframe containing data from the Slisemap object (columns: "X_*", "Y_*", "Z_*", "B_*", "Local loss", ("L_*", "Clusters *")).
    """
    sm = path if isinstance(path, Slisemap) else Slisemap.load(path, "cpu")

    Z = sm.get_Z(rotate=True)
    ss = subsample(Z, max_n) if max_n > 0 and sm.n > max_n else ...

    def preface_names(names: Sequence, preface: str) -> List[str]:
        return [n if n[:2] == preface else preface + n for n in map(str, names)]

    variables = sm.metadata.get_variables(intercept=False)
    variables = preface_names(variables, "X_")
    targets = sm.metadata.get_targets()
    if len(targets) > 1 or targets[0] != "Y":
        targets = preface_names(targets, "Y_")
    predictions = ["Ŷ" + t[1:] for t in targets]
    coefficients = sm.metadata.get_coefficients()
    coefficients = preface_names(coefficients, "B_")
    dimensions = sm.metadata.get_dimensions()
    dimensions = preface_names(dimensions, "Z_")
    rows = sm.metadata.get_rows(fallback=False)
    has_index = True
    if ss is not ...:
        rows = ss if rows is None else np.asarray(rows)[ss]
    elif rows is None:
        has_index = False
        rows = range(sm.n)

    dfs = [
        pd.DataFrame.from_records(sm.metadata.unscale_X()[ss, :], columns=variables),
        pd.DataFrame.from_records(sm.metadata.unscale_Y()[ss, :], columns=targets),
        pd.DataFrame.from_records(Z[ss, :], columns=dimensions),
        pd.DataFrame.from_records(sm.get_B()[ss, :], columns=coefficients),
        pd.DataFrame.from_records(
            sm.metadata.unscale_Y(sm.predict(X=sm._X[ss, :], B=sm._B[ss, :])),
            columns=predictions,
        ),
    ]
    del variables, targets, dimensions, coefficients, predictions
    gc.collect(1)

    L = sm.get_L(X=sm._X[ss, :], Y=sm._Y[ss, :])[ss, :]
    dfs.append(pd.DataFrame({"Local loss": L.diagonal()}))
    if not isinstance(losses, bool) and losses > 0 and losses * 2 < Z.shape[0]:
        sel = subsample(Z[ss, :], losses)
        sel.sort()
        Ln = [f"LT_{rows[i]}" for i in sel]
        dfs.append(pd.DataFrame.from_records(L.T[:, sel], columns=Ln))
    elif losses:
        Ln = [f"L_{i}" for i in rows]
        dfs.append(pd.DataFrame.from_records(L, columns=Ln))
    del L, Z
    gc.collect(1)

    if clusters > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            clusters = {
                f"Clusters {i}": pd.Series(
                    sm.get_model_clusters(i)[0][ss], dtype="category"
                )
                for i in range(2, clusters + 1)
            }
            dfs.append(pd.DataFrame(clusters))

    # Then we create a dataframe to return
    del sm
    gc.collect(1)
    df = pd.concat(dfs, axis=1, copy=False)
    if has_index:
        if index:
            df.index = rows
        else:
            df.insert(0, "item", rows)
    del dfs
    gc.collect(1)
    return df


def _extract_extension(path: Union[str, PathLike]) -> str:
    extension = path if isinstance(path, str) else Path(path).name
    return extension.split(".")[-1]


def load(
    path: Union[Slisemap, pd.DataFrame, str, PathLike],
    extension: Optional[str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load a dataframe or Slisemap object (into a dataframe).

    Args:
        path: Slisemap object, dataframe, path to dataframe, or path to Slisemap object.
        extension: File type (taken from the path if None). Defaults to None.
        **kwargs: Additional args to `slisemap_to_dataframe` when converting Slisemap objects to Dataframes.

    Returns:
        Dataframe (converted from Slisemap object if necessary).
    """
    if isinstance(path, pd.DataFrame):
        return path
    if isinstance(path, Slisemap):
        return slisemap_to_dataframe(path, **kwargs)
    if extension is None:
        extension = _extract_extension(path)
    if extension == "csv":
        return pd.read_csv(path)
    if extension == "parquet" or extension == "pq":
        return pd.read_parquet(path)
    if extension == "json":
        return pd.read_json(path)
    if extension == "feather" or extension == "ft":
        return pd.read_feather(path)
    return slisemap_to_dataframe(path, **kwargs)


def save_dataframe(
    df: pd.DataFrame, path: PathLike, extension: Optional[str] = None
) -> None:
    """Save dataframe to a file.

    Supports csv, json, feather, and parquet.

    Args:
        df: Dataframe.
        path: Path to the file.
        extension: File type (taken from the path if None). Defaults to None.

    Raises:
        NotImplementedError: For unknown file extensions.
    """
    if extension is None:
        extension = _extract_extension(path)
    if "item" not in df.columns and not (
        isinstance(df.index, pd.RangeIndex)
        and df.index.identical(pd.RangeIndex.from_range(range(df.shape[0])))
    ):
        df = df.reset_index().rename(columns={"index": "item"})
    if extension == "csv":
        df.to_csv(path, index=False)
    elif extension == "json":
        df.to_json(path)
    elif extension == "ft" or extension == "feather":
        df.to_feather(path)
    elif extension == "parquet" or extension == "pq":
        df.to_parquet(path)
    else:
        raise NotImplementedError(f"Unknown file format for {extension}")


def get_L_column(df: pd.DataFrame, index: Optional[int] = None) -> Optional[np.ndarray]:
    """Get a column of the L matrix from a `slisemap_to_dataframe`.

    If `df` only contains a partial L matrix, then some values will be `np.nan`.
    If `df` does not contain L, then `None` is returned.

    Args:
        df: Dataframe from `slisemap_to_dataframe`.
        index: Column index. Defaults to None.

    Returns:
        Loss column.
    """
    rows = df.get("item", df.index)
    col = df.get(f"L_{rows[index]}", None)
    if col is not None:
        return col
    lts = {k: i for i, k in enumerate(df) if k[:3] == "LT_"}
    if len(lts) == 0:
        return None
    loss = np.repeat(np.nan, df.shape[0])
    loss[index] = df["Local loss"].iloc[index]
    for i, row in enumerate(rows):
        col = lts.get(f"LT_{row}", -1)
        if col >= 0:
            loss[i] = df.iloc[index, col]
    return loss
