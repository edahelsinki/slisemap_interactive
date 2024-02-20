import numpy as np
import pytest
from slisemap import Slipmap, Slisemap

from slisemap_interactive.load import (
    get_L_column,
    load,
    save_dataframe,
    slipmap_to_dataframe,
    slisemap_to_dataframe,
    subsample,
)


@pytest.fixture(scope="session")
def sm_to_df():
    X = np.random.normal(0, 1, (100, 5))
    Y = np.random.normal(0, 1, 100)
    B0 = np.random.normal(0, 1, (100, 6))
    sm = Slisemap(X, Y, lasso=0.1, B0=B0)
    df = slisemap_to_dataframe(sm, losses=True)
    return sm, df


def test_subsample():
    Z = np.random.normal(0, 1, (20, 2))
    ss1 = subsample(Z, 10)
    assert ss1.size == 10
    assert np.all(ss1 < 20)
    assert np.all(ss1 >= 0)
    assert np.unique(ss1).size == ss1.size
    ss2 = subsample(Z, 10, 0)
    assert ss2.size == 10
    assert np.all(ss2 < 20)
    assert np.all(ss2 >= 0)
    assert np.unique(ss2).size == ss2.size
    ss3 = subsample(Z, 30)
    assert np.allclose(ss3, np.arange(20))


def test_load_slisemap(sm_to_df):
    sm, df = sm_to_df
    assert np.allclose(sm.get_X()[:, 0], df["X_0"])
    assert np.allclose(sm.get_Y()[:, 0], df["Y"])
    assert np.allclose(sm.get_B()[:, 0], df["B_0"])
    assert np.allclose(sm.get_L()[:, 0], df["L_0"])
    assert np.allclose(sm.get_Z(rotate=True)[:, 0], df["Z_0"])
    assert df.shape[0] == sm.n
    assert df.shape[1] == 1 + sm.n + 7 + sm.m + sm.q + sm.o * 2 + sm.d - sm.intercept
    df2 = slisemap_to_dataframe(sm, max_n=80, index=False, losses=False, clusters=(3,))
    assert df2.shape[0] == 80
    sm.metadata.set_rows(range(1, sm.n + 1))
    sm.metadata.set_variables(range(1, sm.m + 1 - sm.intercept))
    sm.metadata.set_targets("test")
    sm.metadata.set_coefficients(sm.metadata.get_variables())
    sm.metadata.set_dimensions("as")
    df3 = slisemap_to_dataframe(sm, losses=10, clusters=None)
    assert all(f"X_{i}" in df3 for i in sm.metadata.get_variables(intercept=False))
    assert all(f"B_{i}" in df3 for i in sm.metadata.get_coefficients())
    assert all(f"Y_{i}" in df3 for i in sm.metadata.get_targets())
    assert all(f"Z_{i}" in df3 for i in sm.metadata.get_dimensions())
    assert np.allclose(df3.index, sm.metadata.get_rows())
    slisemap_to_dataframe(sm, losses=20, max_n=20, clusters=None, index=False)


def test_load_slipmap(sm_to_df):
    sm, dfm = sm_to_df
    sp = Slipmap.convert(sm, prototypes=10)
    dfp = slipmap_to_dataframe(sp, clusters=None, losses=False)
    dfp = slipmap_to_dataframe(sp, clusters=range(6, 7), losses=True)
    for col in dfp.columns:
        if col[0] in ("X", "Y"):
            assert np.allclose(dfm[col], dfp[col][: dfm.shape[0]])
        if col[:3] == "LT_":
            assert np.all(np.isnan(dfp[col][sp.n :]))
            assert np.all(np.isfinite(dfp[col][: sp.n]))
            assert int(col[3:]) >= sp.n
    assert np.sum(np.isnan(get_L_column(dfp, 0))) == sp.n - 1


def test_rec_l(sm_to_df):
    sm, df1 = sm_to_df
    df2 = slisemap_to_dataframe(sm, losses=30, clusters=None)
    for i in range(df1.shape[0]):
        l1 = get_L_column(df1, i)
        l2 = get_L_column(df2, i)
        assert np.all(np.isclose(l1, l2) + np.isnan(l2))


def test_save(sm_to_df, tmp_path):
    _, df = sm_to_df
    df2 = df.copy()
    df2.index = range(1, df.shape[0] + 1)
    for ending in ["csv", "json", "feather", "parquet"]:
        try:
            save_dataframe(df, tmp_path / f"test.{ending}")
            save_dataframe(df, str(tmp_path / f"test.{ending}"))
            df3 = load(tmp_path / f"test.{ending}")
            assert "item" not in df3.columns
            assert np.allclose(df3.to_numpy(), df.to_numpy())
            save_dataframe(df2, str(tmp_path / f"test.{ending}"))
            df3 = load(tmp_path / f"test.{ending}")
            del df3["item"]
            assert np.allclose(df3.to_numpy(), df.to_numpy())
        except ImportError:
            pass
