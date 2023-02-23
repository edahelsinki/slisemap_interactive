from slisemap import Slisemap
import numpy as np
from slisemap_interactive.load import slisemap_to_dataframe


def test_load_slisemap():
    X = np.random.normal(0, 1, (100, 5))
    Y = np.random.normal(0, 1, 100)
    B0 = np.random.normal(0, 1, (100, 6))
    sm = Slisemap(X, Y, lasso=0.1, B0=B0)
    df = slisemap_to_dataframe(sm, losses=True, clusters=8)
    assert np.allclose(X[:, 0], df["X_0"])
    assert np.allclose(Y, df["Y"])
    assert np.allclose(sm.get_B()[:, 0], df["B_0"])
    assert np.allclose(sm.get_L()[:, 0], df["L_0"])
    assert np.allclose(sm.get_Z(rotate=True)[:, 0], df["Z_0"])
    assert df.shape[0] == sm.n
    assert df.shape[1] == 1 + sm.n + 7 + sm.m + sm.q + sm.o + sm.d - sm.intercept
    df2 = slisemap_to_dataframe(sm, max_n=80, losses=False, clusters=3)
    assert df2.shape[0] == 80
    sm.metadata.set_rows(range(1, sm.n + 1))
    sm.metadata.set_variables(range(1, sm.m + 1 - sm.intercept))
    sm.metadata.set_targets("test")
    sm.metadata.set_coefficients(sm.metadata.get_variables())
    sm.metadata.set_dimensions("as")
    df3 = slisemap_to_dataframe(sm, losses=False, clusters=0)
    assert all(f"X_{i}" in df3 for i in sm.metadata.get_variables(intercept=False))
    assert all(f"B_{i}" in df3 for i in sm.metadata.get_coefficients())
    assert all(f"Y_{i}" in df3 for i in sm.metadata.get_targets())
    assert all(f"Z_{i}" in df3 for i in sm.metadata.get_dimensions())
    assert np.allclose(df3.index, sm.metadata.get_rows())
