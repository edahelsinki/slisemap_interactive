from os import PathLike
from typing import Optional, Sequence, Union

import pandas as pd
import numpy as np
from slisemap import Slisemap


def slisemap_to_dataframe(
    path: Union[str, PathLike, Slisemap],
    variables: Optional[Sequence[str]] = None,
    targets: Union[str, Sequence[str], None] = None,
    losses: bool = False,
    clusters: int = 0,
) -> pd.DataFrame:
    """Convert a ``Slisemap`` object to a ``pandas.DataFrame``.

    Args:
        path: Slisemap object or path to a saved slisemap object.
        variables: List of variables names (columns in X). Defaults to None.
        targets: List of target names (columns in Y). Defaults to None.
        losses: Return the loss matrix. Default to False.
        clusters: Return cluster indices (if greater than zero). Defaults to zero.

    Returns:
        A dataframe containing the X, Y, Z, B, fidelity, (L) matrices from the Slisemap object.
    """

    if isinstance(path, Slisemap):
        sm = path
    else:
        sm = Slisemap.load(path, "cpu")

    Z = sm.get_Z(rotate=True)
    X = sm.get_X(intercept=False)
    Y = sm.get_Y()
    B = sm.get_B()
    L = sm.get_L()
    data = [X, Y, Z, B, L.diagonal()[:, None]]
    if losses:
        data.append(L)
    if clusters:
        data.append(sm.get_model_clusters(clusters))
    mat = np.concatenate(data, 1)

    # The following mess is just to enable optional variable and target names
    names = []
    if variables:
        assert len(variables) == X.shape[1]
        names += [f"X_{i}" for i in variables]
    else:
        names += [f"X_{i+1}" for i in range(X.shape[1])]
    if targets:
        if isinstance(targets, str):
            assert Y.shape[1] == 1
            targets = [targets]
        assert len(targets) == Y.shape[1]
        names += [f"Y_{i}" for i in targets]
    elif Y.shape[1] == 1:
        names.append("Y")
    else:
        names += [f"Y_{i+1}" for i in range(Y.shape[1])]
    names += [f"Z_{i+1}" for i in range(Z.shape[1])]
    if variables:
        if sm.intercept:
            variables = variables + ["B_Intercept"]
        if B.shape[1] == len(variables):
            names += [f"B_{i}" for i in variables]
        elif B.shape[1] % len(variables) == 0 and B.shape[1] % len(targets) == 0:
            variables = [f"{t}:{v}" for t in targets for v in variables]
            names += [f"B_{i}" for i in variables[: B.shape[1]]]
        else:
            names += [f"B_{i+1}" for i in range(X.shape[1])]
    else:
        if sm.intercept:
            names += [f"B_{i+1}" for i in range(B.shape[1] - 1)] + ["B_Intercept"]
        else:
            names += [f"B_{i+1}" for i in range(B.shape[1])]
    names.append("Fidelity")
    if losses:
        names += [f"L_{i+1}" for i in range(L.shape[1])]
    if clusters:
        names.append("Cluster")

    # Then we create a dataframe to return
    df = pd.DataFrame.from_records(mat, columns=names)
    return df
