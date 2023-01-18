import numpy as np
from slisemap import Slisemap

from xiplot_slisemap.load import *


def test_load_slisemap():
    X = np.random.normal(0, 1, (10, 5))
    Y = np.random.normal(0, 1, 10)
    sm = Slisemap(X, Y)
    df = slisemap_to_dataframe(sm, losses=True)
    assert np.allclose(X[:, 0], df["X_1"])
    assert np.allclose(Y, df["Y"])
    assert np.allclose(sm.get_B()[:, 0], df["B_1"])
    assert np.allclose(sm.get_L()[:, 0], df["L_1"])
