"""Functions and classes for generating dynamic plots."""

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    get_args,
)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
from dash import ALL, MATCH, Dash, Input, Output, ctx, dcc, html, no_update
from pandas.api.types import is_bool_dtype, is_categorical_dtype, is_object_dtype
from plotly.graph_objects import Figure
from scipy.stats import gaussian_kde

from slisemap_interactive.load import get_L_column

PLOTLY_TEMPLATE = "slisemap_interactive"
DEFAULT_TEMPLATE = "plotly_white+" + PLOTLY_TEMPLATE
pio.templates[PLOTLY_TEMPLATE] = go.layout.Template(
    layout={
        "margin": {"l": 10, "r": 10, "t": 30, "b": 20, "autoexpand": True},
        "uirevision": True,
    }
)


def try_twice(fn: Callable[[], Any], *args: Any, **kwargs: Any) -> Any:
    """Call a function and if it throws an exception retry it once more.

    Args:
        fn: The function to call.
        *args: Arguments to the function `fn`.
        **kwargs: Keyword arguments to the function `fn`.

    Returns:
        The output from the function `fn`.
    """
    try:
        return fn(*args, **kwargs)
    except Exception:
        return fn(*args, **kwargs)


def nested_get(obj: Any, *keys: Any) -> Optional[Any]:
    """Get a value from a nested object.

    Args:
        obj: The nested object.
        *keys: Keys to traverse the nested object.

    Returns:
        The value or None if it does not exist.
    """
    for key in keys:
        if obj is None or len(obj) == 0:
            return None
        obj = obj[key]
    return obj


def first_not_none(
    objects: Sequence[Optional[Any]],
    map: Optional[Callable[[Any], Optional[Any]]] = None,
    *args: Any,
) -> Optional[Any]:
    """Find the first value that is not `None` (with optional mapping function).

    Args:
        objects: List of possible `None` objects.
        map: Transformation function that might return `None`. Defaults to `None`.
        *args: Forwarded to map.

    Returns:
        First (mapped) not-`None` object or `None` if all are `None`.
    """
    for obj in objects:
        if obj is not None:
            if map is None:
                return obj
            else:
                obj = map(obj, *args)
                if obj is not None:
                    return obj
    return None


def is_cluster_or_categorical(df: pd.DataFrame, column: str) -> bool:
    """Check if the column exists in the dataframe and if it is categorical or cluster-like.

    Args:
        df: Dataframe.
        column: Column.

    Returns:
        True if the column exists and should be treated as categorical.
    """
    col = df.get(column, None)
    if col is None:
        return False
    if is_categorical_dtype(col):
        return True
    if is_bool_dtype(col):
        return True
    if "cluster" in column.lower():
        return True
    if is_object_dtype(col) and len(col[:50].unique()) <= 10:  # noqa: SIM102
        if len(col.unique()) <= 10:
            return True
    return False


def get_categories(series: pd.Series) -> Iterable[str]:
    """Get the categories from a `pd.Series`.

    Args:
        series: Categorical or assumed to be categorical column of a dataframe.

    Returns:
        Categories in an index or array.
    """
    if is_categorical_dtype(series):
        return series.cat.categories
    else:
        un = series.unique()
        # un.sort()
        return un


def get_variables(
    df: pd.DataFrame, clusters: bool = False, loss_first: bool = True
) -> List[str]:
    """Get a list of generally plottable column names (skips the L and Z matrices).

    Args:
        df: Slisemap converted to dataframe.
        clusters: Include cluster columns. Defaults to False.
        loss_first: Move the local loss first. Defaults to True.

    Returns:
        list of column names.
    """
    vars = [
        c
        for c in df.columns
        if c[:2] != "L_"
        and c[:3] != "LT_"
        and c[:2] != "Z_"
        and (not clusters or "cluster" not in c.lower())
    ]
    if loss_first:
        vars2 = [v for v in vars if v != "Local loss"]
        if len(vars) - 1 == len(vars2):
            vars = ["Local loss", *vars2]
    return vars


def placeholder_figure(text: str) -> Dict[str, Any]:
    """Display a placeholder text instead of a graph.

    This can be used in a "callback" function when a graph cannot be rendered.

    Args:
        text: Placeholder text.

    Returns:
        Dash figure (to place into a `Output(dcc.Graph.id, "figure")`).
    """
    return {
        "layout": {
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "annotations": [
                {
                    "text": text,
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 28},
                }
            ],
        }
    }


def kde2d(
    x: np.ndarray, y: np.ndarray, binwidth: Union[None, str, float] = None, n: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate a gaussian kernel density grid.

    Args:
        x: Variable 1.
        y: Variable 2.
        binwidth: Bin width or binwidth method. Defaults to None.
        n: Grid size. Defaults to 20.

    Returns:
        x: Grid x.
        y: Grid y.
        z: Grid z (density).
    """
    gx = np.linspace(np.min(x), np.max(x), n)
    gy = np.linspace(np.min(y), np.max(y), n)
    gx, gy = np.meshgrid(gx, gy)
    gx = gx.ravel()
    gy = gy.ravel()
    gauss = gaussian_kde(np.stack((x, y), 0), binwidth)
    return gx, gy, gauss(np.stack((gx, gy), 0))


class DataCache(dict):
    """Class for holding datasets."""

    def add_data(self, df: pd.DataFrame) -> int:
        """Add a dataset to the cache.

        This function checks for and reuses duplicate datasets.

        Args:
            df: Dataset

        Returns:
            The key to find the dataset in the cache
        """
        for key, value in self.items():
            if value.equals(df):
                return key
        key = np.random.randint(1, np.iinfo(np.int32).max)
        while key in self:
            key = np.random.randint(1, np.iinfo(np.int32).max)
        self[key] = df.copy()
        return key


class JitterSlider(html.Div):
    """Slider for jitter."""

    def __init__(
        self,
        data: Optional[int] = None,
        controls: str = "default",
        scale: float = 0.2,
        steps: int = 5,
        id: Optional[Any] = None,
        value: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Create a jitter slider."""
        values = np.linspace(0.0, scale, steps)
        marks = {0: "No jitter"}
        for v in values[1:]:
            marks[v] = f"{v:g}"
        if id is None:
            assert data is not None and controls is not None
            id = self.generate_id(data, controls)
        slider = dcc.Slider(id=id, min=0.0, max=scale, marks=marks, value=value)
        super().__init__(children=[slider], **kwargs)

    @classmethod
    def generate_id(cls, data: int, controls: str) -> Dict[str, Any]:
        """Generate dash id."""
        return {"type": cls.__name__, "data": data, "controls": controls}


class VariableDropdown(dcc.Dropdown):
    """Dropdown for selecting variable."""

    def __init__(
        self,
        df: pd.DataFrame,
        data: Optional[int] = None,
        controls: str = "default",
        id: Optional[Any] = None,
        value: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Create variable dropdown."""
        vars = get_variables(df)
        if value is None or value not in vars:
            value = vars[0]
        if id is None:
            assert data is not None and controls is not None
            id = self.generate_id(data, controls)
        super().__init__(vars, value, id=id, clearable=False, **kwargs)

    @classmethod
    def generate_id(cls, data: int, controls: str) -> Dict[str, Any]:
        """Generate dash id."""
        return {"type": cls.__name__, "data": data, "controls": controls}


class ClusterDropdown(dcc.Dropdown):
    """Dropdown for selecting cluster."""

    def __init__(
        self,
        df: pd.DataFrame,
        data: Optional[int] = None,
        controls: str = "default",
        id: Optional[Any] = None,
        value: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Create cluster dropdown."""
        clusters = [c for c in df.columns if is_cluster_or_categorical(df, c)]
        if id is None:
            assert data is not None and controls is not None
            id = self.generate_id(data, controls)
        super().__init__(
            clusters, value, id=id, clearable=True, placeholder="No clusters", **kwargs
        )

    @classmethod
    def generate_id(cls, data: int, controls: str) -> Dict[str, Any]:
        """Generate dash id."""
        return {"type": cls.__name__, "data": data, "controls": controls}


class DensityTypeDropdown(dcc.Dropdown):
    """Dropdown for selecting density plot type."""

    def __init__(
        self,
        data: Optional[int] = None,
        controls: str = "default",
        id: Optional[Any] = None,
        value: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Create density type dropdown."""
        if id is None:
            assert data is not None and controls is not None
            id = self.generate_id(data, controls)
        options = get_args(DistributionPlot.PLOT_TYPE_OPTIONS)
        if value is None or value not in options:
            value = options[0]
        super().__init__(options, value, id=id, clearable=False, **kwargs)

    @classmethod
    def generate_id(cls, data: int, controls: str) -> Dict[str, Any]:
        """Generate dash id."""
        return {"type": cls.__name__, "data": data, "controls": controls}


class BarGroupingDropdown(dcc.Dropdown):
    """Dropdown for selecting grouping."""

    def __init__(
        self,
        data: Optional[int] = None,
        controls: str = "default",
        id: Optional[Any] = None,
        value: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Create grouping dropdown."""
        if id is None:
            assert data is not None and controls is not None
            id = self.generate_id(data, controls)
        options = get_args(ModelBarPlot.GROUPING_OPTIONS)
        if value is None or value not in options:
            value = options[0]
        super().__init__(options, value, id=id, clearable=False, **kwargs)

    @classmethod
    def generate_id(cls, data: int, controls: str) -> Dict[str, Any]:
        """Generate dash id."""
        return {"type": cls.__name__, "data": data, "controls": controls}


class PredictionDropdown(dcc.Dropdown):
    """Dropdown for selecting prediction."""

    def __init__(
        self,
        df: pd.DataFrame,
        data: Optional[int] = None,
        controls: str = "default",
        id: Optional[Any] = None,
        value: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Create prediction dropdown."""
        vars = [c for c in df.columns if c[0] == "Ŷ"]
        if len(vars) == 0:
            value = None
        elif value is None or value not in vars:
            value = vars[0]
        if id is None:
            assert data is not None and controls is not None
            id = self.generate_id(data, controls)
        super().__init__(
            vars, value, id=id, clearable=False, disabled=len(vars) < 2, **kwargs
        )

    @classmethod
    def generate_id(cls, data: int, controls: str) -> Dict[str, Any]:
        """Generate dash id."""
        return {"type": cls.__name__, "data": data, "controls": controls}


class ContourCheckbox(dcc.Checklist):
    """Checkbox for contours."""

    def __init__(
        self,
        data: Optional[int] = None,
        controls: str = "default",
        id: Optional[Any] = None,
        value: bool = False,
        **kwargs: Any,
    ) -> None:
        """Create contour checkbox."""
        if id is None:
            assert data is not None and controls is not None
            id = self.generate_id(data, controls)
        super().__init__(["Contours"], ["Contours"] if value else [], id=id, **kwargs)

    @classmethod
    def generate_id(cls, data: int, controls: str) -> Dict[str, Any]:
        """Generate dash id."""
        return {"type": cls.__name__, "data": data, "controls": controls}


class EmbeddingPlot(dcc.Graph):
    """Plot with 2D embedding."""

    def __init__(
        self,
        data: int,
        controls: str = "default",
        hover: str = "default",
        **kwargs: Any,
    ) -> None:
        """Create embedding plot."""
        super().__init__(
            id=self.generate_id(data, controls, hover), clear_on_unhover=True, **kwargs
        )

    @classmethod
    def generate_id(cls, data: int, controls: str, hover: str) -> Dict[str, Any]:
        """Generate dash id."""
        return {
            "type": cls.__name__,
            "data": data,
            "controls": controls,
            "hover": hover,
            "hover_input": 1,
        }

    @classmethod
    def get_hover_index(cls, hover_data: Any) -> Optional[int]:
        """Get the index of the current hover over point."""
        return nested_get(hover_data, "points", 0, "customdata", 0)

    @classmethod
    def register_callbacks(cls, app: Dash, data: DataCache) -> None:
        """Register Dash callbacks."""

        @app.callback(
            Output(cls.generate_id(MATCH, MATCH, MATCH), "figure"),
            Input(JitterSlider.generate_id(MATCH, MATCH), "value"),
            Input(VariableDropdown.generate_id(MATCH, MATCH), "value"),
            Input(ContourCheckbox.generate_id(MATCH, MATCH), "value"),
            Input(ClusterDropdown.generate_id(MATCH, MATCH), "value"),
            Input(HoverData.generate_id(MATCH, MATCH), "data"),
        )
        def callback(jitter, variable, contour, cluster, hover) -> Figure:  # noqa: ANN001
            data_key = ctx.triggered_id["data"]
            df = data[data_key]
            dimensions = filter(lambda c: c[:2] == "Z_", df.columns)
            x = next(dimensions)
            y = next(dimensions)
            if cluster in df.columns:
                variable = cluster
            return cls.plot(df, x, y, variable, contour, jitter, hover, seed=data_key)

    @staticmethod
    def plot(
        df: pd.DataFrame,
        x: str,
        y: str,
        variable: str,
        contours: bool = True,
        jitter: float = 0.0,
        hover: Optional[int] = None,
        seed: int = 42,
        template: str = DEFAULT_TEMPLATE,
    ) -> Figure:
        """Create the plot."""

        def dfmod(var: str) -> pd.DataFrame:
            df2 = pd.DataFrame(
                {x: df[x], y: df[y], var: df[var], "index": np.arange(df.shape[0])}
            )
            if jitter > 0:
                prng = np.random.default_rng(seed)
                df2[x] += prng.normal(0, jitter, df.shape[0])
                df2[y] += prng.normal(0, jitter, df.shape[0])
            return df2

        fig = None
        if is_cluster_or_categorical(df, variable):
            cats = get_categories(df[variable])
            df2 = dfmod(variable)
            df2[variable] = df2[variable].astype("category")
            fig = px.scatter(
                df2,
                x=x,
                y=y,
                color=variable,
                color_discrete_sequence=px.colors.qualitative.Plotly,
                symbol=variable,
                category_orders={variable: cats},
                title="Embedding",
                custom_data=["index"],
            )
            fig.update_traces(hovertemplate=None, hoverinfo="none")
            ll = False
        else:
            ll = variable == "Local loss"
        if fig is None and ll and hover is not None:
            losses = get_L_column(df, hover)
            if losses is not None:
                loss_cols = [c for c in df.columns if c[:2] == "L_" or c[:3] == "LT_"]
                lrange = (
                    df[loss_cols].abs().min().quantile(0.05) * 0.9,
                    df[loss_cols].abs().max().quantile(0.95) * 1.1,
                )
                df2 = dfmod(variable)
                df2[variable] = losses
                fig = px.scatter(
                    df2,
                    x=x,
                    y=y,
                    color=variable,
                    title=f"Alternative locations for item: {df.get('item', df.index)[hover]}",
                    opacity=np.isfinite(losses) * 0.8,
                    color_continuous_scale="Viridis_r",
                    labels={variable: "Local loss&nbsp;"},
                    custom_data=["index"],
                    range_color=lrange,
                )
        if fig is None:
            df2 = dfmod(variable)
            fig = px.scatter(
                df2,
                x=x,
                y=y,
                color=variable,
                color_continuous_scale="Plasma_r",
                title="Embedding",
                opacity=0.8,
                labels={variable: "Local loss&nbsp;"} if ll else None,
                custom_data=["index"],
            )
        if ll:
            fig.update_traces(hovertemplate=None, hoverinfo="none")
        if contours:
            kdex, kdey, kdez = kde2d(df[x], df[y], 0.2, 40)
            fig.add_contour(
                x=kdex,
                y=kdey,
                z=kdez,
                contours_coloring="none",
                ncontours=7,
                showlegend=False,
                hoverinfo="skip",
                hovertemplate=None,
                name="Contours",
                line_color="grey",
                line_width=1,
            )
        if hover is not None:
            trace = px.scatter(df2.iloc[[hover]], x=x, y=y).update_traces(
                hoverinfo="skip",
                hovertemplate=None,
                marker={
                    "size": 15,
                    "color": "rgba(0,0,0,0)",
                    "line": {"width": 1, "color": "black"},
                },
            )
            fig.add_traces(trace.data)
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(template=template, xaxis_title=None, yaxis_title=None)
        return fig


class ModelMatrixPlot(dcc.Graph):
    """Heatmap plot for the coefficient matrix."""

    def __init__(
        self,
        data: int,
        controls: str = "default",
        hover: str = "default",
        **kwargs: Any,
    ) -> None:
        """Create model matrix plot."""
        super().__init__(
            id=self.generate_id(data, controls, hover), clear_on_unhover=True, **kwargs
        )

    @classmethod
    def generate_id(cls, data: int, controls: str, hover: str) -> Dict[str, Any]:
        """Generate dash id."""
        return {
            "type": cls.__name__,
            "data": data,
            "controls": controls,
            "hover": hover,
            "hover_input": 1,
        }

    @classmethod
    def get_hover_index(cls, hover_data: Any) -> Optional[int]:
        """Get the index of the current hover over point."""
        hover = nested_get(hover_data, "points", 0, "x")
        return int(hover) if hover is not None else None

    @classmethod
    def register_callbacks(cls, app: Dash, data: DataCache) -> None:
        """Register Dash callbacks."""

        @app.callback(
            Output(cls.generate_id(MATCH, MATCH, MATCH), "figure"),
            Input(HoverData.generate_id(MATCH, MATCH), "data"),
        )
        def callback(hover: Optional[int]) -> Figure:
            data_key = ctx.triggered_id["data"]
            df = data[data_key]
            zs0 = next(filter(lambda c: c[:2] == "Z_", df.columns))
            bs = [c for c in df.columns if c[:2] == "B_"]
            return cls.plot(df, bs, zs0, hover)

    @staticmethod
    def plot(
        df: pd.DataFrame,
        coefficients: Sequence[str],
        sort_by: Optional[str] = None,
        hover: Optional[int] = None,
        template: str = DEFAULT_TEMPLATE,
    ) -> Figure:
        """Create the plot."""
        if sort_by is None:
            order_to_sorted = np.arange(df.shape[0])
        else:
            order_to_sorted = df[sort_by].to_numpy().argsort()
        sorted_to_string = [str(i) for i in order_to_sorted]
        B_mat = df[coefficients].to_numpy()[order_to_sorted, :].T

        fig = px.imshow(
            B_mat,
            color_continuous_midpoint=0,
            aspect="auto",
            labels={"color": "Coefficient", "x": "Data items sorted left to right"},
            title="Local models",
            color_continuous_scale="RdBu",
            y=coefficients,
            x=sorted_to_string,
            template=template,
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_traces(hovertemplate="%{y} = %{z}<extra></extra>")
        if hover is not None:
            fig.add_vline(x=np.nonzero(order_to_sorted == hover)[0][0])
        return fig


class ModelBarPlot(dcc.Graph):
    """Barplot for the local model coefficients."""

    GROUPING_OPTIONS = Literal["Variables", "Clusters"]

    def __init__(
        self,
        data: int,
        controls: str = "default",
        hover: str = "default",
        **kwargs: Any,
    ) -> None:
        """Create the model bar plot."""
        super().__init__(id=self.generate_id(data, controls, hover), **kwargs)

    @classmethod
    def generate_id(cls, data: int, controls: str, hover: str) -> Dict[str, Any]:
        """Generate dash id."""
        return {
            "type": cls.__name__,
            "data": data,
            "controls": controls,
            "hover": hover,
        }

    @classmethod
    def register_callbacks(cls, app: Dash, data: DataCache) -> None:
        """Register Dash callbacks."""

        @app.callback(
            Output(cls.generate_id(MATCH, MATCH, MATCH), "figure"),
            Input(ClusterDropdown.generate_id(MATCH, MATCH), "value"),
            Input(BarGroupingDropdown.generate_id(MATCH, MATCH), "value"),
            Input(HoverData.generate_id(MATCH, MATCH), "data"),
        )
        def callback(
            cluster: Optional[str], grouping: cls.GROUPING_OPTIONS, hover: Optional[int]
        ) -> Figure:
            data_key = ctx.triggered_id["data"]
            df = data[data_key]
            coefficients = [c for c in df.columns if c[:2] == "B_"]
            return try_twice(
                lambda: cls.plot(df, coefficients, cluster, grouping, hover)
            )

        @app.callback(
            Output(BarGroupingDropdown.generate_id(MATCH, MATCH), "disabled"),
            Input(ClusterDropdown.generate_id(MATCH, MATCH), "value"),
        )
        def callback_disabled(cluster: Optional[str]) -> bool:
            return cluster is None

    @staticmethod
    def plot(
        df: pd.DataFrame,
        coefficients: Sequence[str],
        cluster: Optional[str] = None,
        grouping: GROUPING_OPTIONS = "Variables",
        hover: Optional[int] = None,
        template: str = DEFAULT_TEMPLATE,
    ) -> Figure:
        """Create the plot."""
        coefficient_range = df[coefficients].abs().quantile(0.95).max() * 1.1
        if hover is not None:
            fig = px.bar(
                pd.DataFrame(df[coefficients].iloc[hover]),
                range_y=(-coefficient_range, coefficient_range),
                color=coefficients,
                color_discrete_sequence=px.colors.qualitative.Plotly,
                title=f"Local model for item: {df.get('item', df.index)[hover]}",
            )
            fig.update_layout(showlegend=False)
        elif is_cluster_or_categorical(df, cluster):
            cats = get_categories(df[cluster])
            df2 = (
                df.groupby([cluster])[coefficients]
                .aggregate(["mean", "std"])
                .stack(level=0)
                .reset_index()
                .rename(columns={"level_1": "Coefficients"})
            )
            df2[cluster] = df2[cluster].astype("category")
            facet = grouping == "Clusters"
            fig = px.bar(
                df2,
                x="Coefficients",
                y="mean",
                error_y="std",
                color=cluster,
                category_orders={cluster: cats},
                color_discrete_sequence=px.colors.qualitative.Plotly,
                range_y=(-coefficient_range, coefficient_range),
                title="Local models",
                barmode="relative" if facet else "group",
                facet_col=cluster if facet else None,
            )
            fig.update_annotations(visible=False)  # Remove facet labels
        else:
            fig = px.bar(
                df[coefficients].aggregate(["mean", "std"]).T,
                y="mean",
                error_y="std",
                color=coefficients,
                color_discrete_sequence=px.colors.qualitative.Plotly,
                range_y=(-coefficient_range, coefficient_range),
                title="Mean local model",
            )
            fig.update_layout(showlegend=False)
        fig.update_xaxes(title=None)
        fig.update_layout(template=template, yaxis_title="Coefficient")
        fig.update_traces(hovertemplate="%{x} = %{y}<extra></extra>")
        return fig


class DistributionPlot(dcc.Graph):
    """Distribution plot for the data and models."""

    def __init__(
        self,
        data: int,
        controls: str = "default",
        hover: str = "default",
        **kwargs: Any,
    ) -> None:
        """Create the distribution plot."""
        super().__init__(id=self.generate_id(data, controls, hover), **kwargs)

    @classmethod
    def generate_id(cls, data: int, controls: str, hover: str) -> Dict[str, Any]:
        """Generate dash id."""
        return {
            "type": cls.__name__,
            "data": data,
            "controls": controls,
            "hover": hover,
        }

    @classmethod
    def register_callbacks(cls, app: Dash, data: DataCache) -> None:
        """Register Dash callbacks."""

        @app.callback(
            Output(cls.generate_id(MATCH, MATCH, MATCH), "figure"),
            Input(VariableDropdown.generate_id(MATCH, MATCH), "value"),
            Input(ClusterDropdown.generate_id(MATCH, MATCH), "value"),
            Input(DensityTypeDropdown.generate_id(MATCH, MATCH), "value"),
            Input(HoverData.generate_id(MATCH, MATCH), "data"),
        )
        def callback(variable, cluster, histogram, hover) -> Figure:  # noqa: ANN001
            data_key = ctx.triggered_id["data"]
            df = data[data_key]
            return try_twice(cls.plot, df, variable, histogram, cluster, hover)

    PLOT_TYPE_OPTIONS = Literal["Histogram", "Density"]

    @staticmethod
    def plot(
        df: pd.DataFrame,
        variable: str,
        plot_type: PLOT_TYPE_OPTIONS = "Histogram",
        cluster: Optional[str] = None,
        hover: Optional[int] = None,
        template: str = DEFAULT_TEMPLATE,
    ) -> Figure:
        """Create the plot."""
        if is_cluster_or_categorical(df, cluster):
            cats = get_categories(df[cluster])
            if plot_type == "Histogram":
                fig = px.histogram(
                    df,
                    variable,
                    color=cluster,
                    color_discrete_sequence=px.colors.qualitative.Plotly,
                    title=f"Histogram of {variable}",
                    marginal="violin",
                    category_orders={cluster: cats},
                )
            else:
                df2 = df.groupby(cluster)[variable]
                data = [df2.get_group(g) for g in cats]
                clusters = [str(g) for g in cats]
                colors = px.colors.qualitative.Plotly
                colors = colors * ((len(clusters) - 1) // len(colors) + 1)
                filter = [not np.allclose(i[0], i) for i in df2.groups.values()]
                if not all(filter):
                    data = [d for d, f in zip(data, filter) if f]
                    colors = [c for c, f in zip(colors, filter) if f]
                    clusters = [c for c, f in zip(clusters, filter) if f]
                fig = ff.create_distplot(data, clusters, show_hist=False, colors=colors)
                fig.update_layout(
                    title=f"Density plot for {variable}",
                    legend={"title": cluster, "traceorder": "normal"},
                )
            if len(cats) < 4:
                fig.layout.yaxis.domain = [0.21, 1]
                fig.layout.yaxis2.domain = [0, 0.19]
            else:
                fig.layout.yaxis.domain = [0.31, 1]
                fig.layout.yaxis2.domain = [0, 0.29]
            fig.layout.yaxis.showgrid = True
            fig.layout.yaxis2.showgrid = True
            fig.layout.yaxis.showticklabels = False
            fig.layout.yaxis2.showticklabels = True
        else:
            if plot_type == "Histogram":
                fig = px.histogram(df, variable, title=f"Histogram of {variable}")
            else:
                fig = ff.create_distplot([df[variable]], [variable], show_hist=False)
                fig.layout.yaxis.domain = [0.21, 1]
                fig.layout.yaxis2.domain = [0, 0.19]
                fig.update_layout(
                    showlegend=False, title=f"Density plot for {variable}"
                )
        if hover is not None:
            fig.add_vline(x=df[variable].iloc[hover])
        fig.update_layout(template=template, xaxis_title=None, yaxis_title=None)
        return fig


class LinearTerms(dcc.Graph):
    """Plot the local model coefficients times the variable values in a barplot.

    This plot assumes that the variables are scaled and the coefficients are linear.
    """

    def __init__(
        self,
        data: int,
        controls: str = "default",
        hover: str = "default",
        **kwargs: Any,
    ) -> None:
        """Create the bar plot."""
        super().__init__(id=self.generate_id(data, controls, hover), **kwargs)

    @classmethod
    def generate_id(cls, data: int, controls: str, hover: str) -> Dict[str, Any]:
        """Generate dash id."""
        return {
            "type": cls.__name__,
            "data": data,
            "controls": controls,
            "hover": hover,
        }

    @classmethod
    def register_callbacks(cls, app: Dash, data: DataCache) -> None:
        """Register Dash callbacks."""

        @app.callback(
            Output(cls.generate_id(MATCH, MATCH, MATCH), "figure"),
            Input(PredictionDropdown.generate_id(MATCH, MATCH), "value"),
            Input(HoverData.generate_id(MATCH, MATCH), "data"),
        )
        def callback(pred: str, hover: Optional[int]) -> Figure:
            data_key = ctx.triggered_id["data"]
            df = data[data_key]
            return try_twice(cls.plot, df, pred, hover)

    @staticmethod
    def plot(
        df: pd.DataFrame,
        pred: str,
        hover: Optional[int] = None,
        decimals: int = 3,
        template: str = DEFAULT_TEMPLATE,
    ) -> Figure:
        """Create the plot."""
        if hover is None:
            return no_update
        Xs = [c for c in df.columns if c[0] == "X"]
        Bs = [c for c in df.columns if c[0] == "B"]
        intercept = "Intercept" in Bs[-1]
        if len(pred) > 2 and len(Bs) > len(Xs) + intercept:
            Bs = [c for c in Bs if pred[2:] in c]
        xrow = df[Xs].iloc[hover]
        brow = df[Bs].iloc[hover]
        if len(Xs) == len(Bs):
            pass
        elif len(Xs) + 1 == len(Bs) and intercept:
            xrow["X_Intercept"] = 1.0
        else:
            return placeholder_figure(
                f"Could not match variables to coefficients for {pred}"
            )
        pred2 = "Y" + pred[1:]
        if pred2 in df.columns:
            y = df[[pred, pred2]].iloc[hover].to_numpy()
            target = True
        else:
            y = df[pred].iloc[hover]
            target = False
        vars = [c[2:] for c in xrow.index]
        xrow = xrow.to_numpy()
        brow = brow.to_numpy()
        terms = brow * xrow
        xdec = int(np.max(np.log(np.abs(xrow) + 1e-8)) // np.log(10))
        xdec = decimals - min(decimals - 1, max(0, xdec))
        bdec = int(np.max(np.log(np.abs(brow) + 1e-8)) // np.log(10))
        bdec = decimals - min(decimals - 1, max(0, bdec))
        ydec = int(np.max(np.log(np.abs(y) + 1e-8)) // np.log(10))
        ydec = decimals - min(decimals - 1, max(0, ydec))
        tdec = int(np.max(np.log(np.abs(terms) + 1e-8)) // np.log(10))
        tdec = decimals - min(decimals - 1, max(0, tdec))
        text = [
            f"X × B = {x:.{xdec}g} × {b:.{bdec}g} = {i:.{tdec}g}"  # noqa: RUF001
            for x, b, i in zip(xrow, brow, terms)
        ]
        xmax = np.max(np.abs(terms)) * 1.01
        df2 = pd.DataFrame(
            {
                "Variable": vars,
                "Value": xrow,
                "Coefficient": brow,
                "text": text,
                "sign": np.sign(terms),
                "Term": terms,
            }
        )
        fig = px.bar(
            df2.iloc[::-1, :],
            x="Term",
            y="Variable",
            color="sign",
            text="text",
            hover_data=["Value", "Coefficient"],
            color_continuous_scale=["orange", "purple"],
            range_x=(-xmax, xmax),
            title=f"Linear terms for item: {df.get('item', df.index)[hover]}",
            template=template,
        )
        if target:
            xax = f"Prediction: {pred} = {y[0]:.{ydec}g},   Target: {pred2} = {y[1]:.{ydec}g}"
        else:
            xax = f"Prediction: {pred} = {y:.{ydec}g}"
        fig.update_layout(
            xaxis_title=xax, yaxis_title=None, coloraxis_showscale=False, hovermode="y"
        )
        fig.update_traces(
            hovertemplate="<b>%{y}</b><br>Value=%{customdata[0]}<br>Coefficient=%{customdata[1]}<br>Term=%{x}<extra></extra>"
        )
        return fig


class HoverData(dcc.Store):
    """Data store for the hover index."""

    def __init__(self, data: int, hover: str = "default", **kwargs: Any) -> None:
        """Create the data store."""
        super().__init__(
            id=self.generate_id(data, hover), data=None, storage_type="memory", **kwargs
        )

    @classmethod
    def generate_id(cls, data: int, hover: str) -> Dict[str, Any]:
        """Generate dash id."""
        return {"type": cls.__name__, "data": data, "hover": hover}

    @classmethod
    def register_callbacks(cls, app: Dash, data: Optional[DataCache] = None) -> None:
        """Register Dash callbacks."""
        input = EmbeddingPlot.generate_id(MATCH, ALL, MATCH)
        input["type"] = ALL

        @app.callback(
            Output(HoverData.generate_id(MATCH, MATCH), "data"),
            Input(input, "hoverData"),
            prevent_initial_call=True,
        )
        def hover_callback(inputs: Any) -> Optional[int]:
            tt = ctx.triggered_id["type"]
            if tt == "EmbeddingPlot":
                return first_not_none(inputs, EmbeddingPlot.get_hover_index)
            elif tt == "ModelMatrixPlot":
                return first_not_none(inputs, ModelMatrixPlot.get_hover_index)
            else:
                return first_not_none(inputs, nested_get, "points", 0, "customdata", 0)
            return None
