import pandas as pd
import pytest

from slisemap_interactive.plots import (
    BarGroupingDropdown,
    ClusterDropdown,
    ContourCheckbox,
    DataCache,
    DensityTypeDropdown,
    DistributionPlot,
    EmbeddingPlot,
    HoverData,
    JitterSlider,
    ModelBarPlot,
    ModelMatrixPlot,
    VariableDropdown,
    first_not_none,
    nested_get,
)


@pytest.fixture(scope="session")
def dataframe():
    df = pd.DataFrame()
    df["cls"] = pd.Categorical([1, 2, 3, 1, 2, 3])
    df["Local loss"] = [0.1, 0.2, 0.3, 0.3, 0.2, 0.1]
    df["L_0"] = df["L_1"] = df["L_2"] = [0.3, 0.2, 0.1, 0.1, 0.2, 0.3]
    df["B_0"] = df["B_1"] = df["B_2"] = [0.3, 0.2, 0.1, 0.3, 0.2, 0.1]
    df["X_0"] = df["X_1"] = [1, 2, 3, 1, 2, 3]
    df["Y_0"] = [4, 6, -2, -4, 4, 0]
    df["Z_0"] = [3, 2, 3, 3, 1, 3]
    df["Z_1"] = [1, 2, 3, 3, 1, 2]
    return df


def test_nested_get():
    assert nested_get({"a": [{"b": [1]}]}, "a") == [{"b": [1]}]
    assert nested_get({"a": [{"b": [1]}]}, "a", 0) == {"b": [1]}
    assert nested_get({"a": [{"b": [1]}]}, "a", 0, "b") == [1]
    assert nested_get({"a": [{"b": [1]}]}, "a", 0, "b", 0) == 1
    assert nested_get({"a": [{"b": None}]}, "a", 0, "b", 0) is None
    assert nested_get({"a": [{"b": []}]}, "a", 0, "b", 0) is None


def test_datacache(dataframe):
    dc = DataCache()
    key = dc.add_data(dataframe)
    assert key == dc.add_data(dataframe)
    assert dc[key].equals(dataframe)
    df2 = dataframe.copy()
    df2["test"] = 1
    assert key != dc.add_data(df2)


def test_first_not_none():
    assert first_not_none([]) is None
    assert first_not_none([None]) is None
    assert first_not_none([1]) == 1
    assert first_not_none([1, 2]) == 1
    assert first_not_none([None, 2]) == 2
    assert first_not_none([3, 2], lambda x: x**2) == 9
    assert first_not_none([3, 2], lambda x: None) is None
    assert first_not_none([3, 2], lambda x: None if x > 2 else 2) == 2


def test_embedding(dataframe):
    EmbeddingPlot(0)
    VariableDropdown(dataframe, id="")
    ContourCheckbox(id="")
    JitterSlider(id="")
    ClusterDropdown(dataframe, id="")
    assert EmbeddingPlot.get_hover_index(None) is None
    assert EmbeddingPlot.get_hover_index({"points": [{"customdata": [1]}]}) == 1
    EmbeddingPlot.plot(dataframe, "Z_0", "Z_1", "Y_0", True, 0.2, 2)
    EmbeddingPlot.plot(dataframe, "Z_0", "Z_1", "cls", False, 0.2, 2)
    EmbeddingPlot.plot(dataframe, "Z_0", "Z_1", "cls", True, 0.0, None)
    EmbeddingPlot.plot(dataframe, "Z_0", "Z_1", "Local loss", True, 0.0, 2)
    EmbeddingPlot.plot(dataframe, "Z_0", "Z_1", "Local loss", False, 0.2, None)


def test_matrix(dataframe):
    ModelMatrixPlot(0)
    assert ModelMatrixPlot.get_hover_index(None) is None
    assert ModelMatrixPlot.get_hover_index({"points": [{"x": 1}]}) == 1
    ModelMatrixPlot.plot(dataframe, ["B_0", "B_1", "B_2"], None, 2)
    ModelMatrixPlot.plot(dataframe, ["B_0", "B_1", "B_2"], None, None)
    ModelMatrixPlot.plot(dataframe, ["B_0", "B_1", "B_2"], "Z_0", 3)
    ModelMatrixPlot.plot(dataframe, ["B_0", "B_1", "B_2"], "Z_0", None)


def test_bar(dataframe):
    ModelBarPlot(0)
    BarGroupingDropdown(id="")
    ClusterDropdown(dataframe, id="")
    ModelBarPlot.plot(dataframe, ["B_0", "B_1", "B_2"], None, "Variables", 2)
    ModelBarPlot.plot(dataframe, ["B_0", "B_1", "B_2"], None, "Variables", None)
    ModelBarPlot.plot(dataframe, ["B_0", "B_1", "B_2"], "No Cluster", "Variables", 2)
    ModelBarPlot.plot(dataframe, ["B_0", "B_1", "B_2"], "No Cluster", "Variables", None)
    ModelBarPlot.plot(dataframe, ["B_0", "B_1", "B_2"], "cls", "Variables", None)
    ModelBarPlot.plot(dataframe, ["B_0", "B_1", "B_2"], "cls", "Clusters", None)
    ModelBarPlot.plot(dataframe, ["B_0", "B_1", "B_2"], "cls", "Variables", 3)
    ModelBarPlot.plot(dataframe, ["B_0", "B_1", "B_2"], "cls", "Clusters", 3)


def test_dist(dataframe):
    DistributionPlot(0)
    VariableDropdown(dataframe, id="")
    DensityTypeDropdown(id="")
    ClusterDropdown(dataframe, id="")
    DistributionPlot.plot(dataframe, "Y_0", "Histogram", None, 2)
    DistributionPlot.plot(dataframe, "Y_0", "Histogram", "cls", 2)
    DistributionPlot.plot(dataframe, "Y_0", "Density", None, 2)
    DistributionPlot.plot(dataframe, "Y_0", "Density", "cls", 2)


def test_hoverdata():
    HoverData(0)
