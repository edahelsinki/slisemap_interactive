"""
Functions for generating dynamic plots.
"""

from typing import Any, Callable, Dict, Literal, Optional, Sequence, get_args

from dash import Dash, html, dcc, Input, Output, ctx, MATCH, ALL
import plotly.express as px
import plotly.figure_factory as ff
from plotly.graph_objects import Figure
import pandas as pd
import numpy as np
from pandas.api.types import is_categorical_dtype

DEFAULT_FIG_LAYOUT = dict(
    margin=dict(l=10, r=10, t=30, b=20, autoexpand=True),
    template="plotly_white",
    uirevision=True,
)


def try_twice(fn: Callable[[], Any]) -> Any:
    """Call a function and if it throws an exception retry it once more.

    Args:
        fn: The function to call

    Returns:
        The output from the function.
    """
    try:
        return fn()
    except:
        return fn()


def nested_get(obj: Any, *keys) -> Optional[Any]:
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
    *args,
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
    def __init__(
        self,
        data: int,
        scale: float = 0.2,
        steps: int = 5,
        controls: str = "default",
        id: Optional[Any] = None,
        **kwargs,
    ):
        values = np.linspace(0.0, scale, steps)
        marks = {0: "No jitter"}
        for v in values[1:]:
            marks[v] = f"{v:g}"
        if id is None:
            id = self.generate_id(data, controls)
        slider = dcc.Slider(id=id, min=0.0, max=scale, marks=marks, value=0.0)
        super().__init__(children=[slider], **kwargs)

    @classmethod
    def generate_id(cls, data: int, controls: str) -> Dict[str, Any]:
        return {"type": cls.__name__, "data": data, "controls": controls}


class VariableDropdown(dcc.Dropdown):
    def __init__(
        self,
        data: int,
        df: pd.DataFrame,
        controls: str = "default",
        id: Optional[Any] = None,
        **kwargs,
    ):
        vars = ["Local loss"] + [c for c in df.columns if c[0] in ("X", "Y", "B")]
        if id is None:
            id = self.generate_id(data, controls)
        super().__init__(vars, vars[0], id=id, clearable=False, **kwargs)

    @classmethod
    def generate_id(cls, data: int, controls: str) -> Dict[str, Any]:
        return {"type": cls.__name__, "data": data, "controls": controls}


class ClusterDropdown(dcc.Dropdown):
    def __init__(
        self,
        data: int,
        df: pd.DataFrame,
        controls: str = "default",
        id: Optional[Any] = None,
        **kwargs,
    ):
        clusters = ["No clusters"]
        while clusters[0] in df.columns:
            clusters[0] += "_"
        clusters.extend(c for c in df.columns if is_categorical_dtype(df[c]))
        if id is None:
            id = self.generate_id(data, controls)
        super().__init__(clusters, clusters[0], id=id, clearable=False, **kwargs)

    @classmethod
    def generate_id(cls, data: int, controls: str) -> Dict[str, Any]:
        return {"type": cls.__name__, "data": data, "controls": controls}


class HistogramDropdown(dcc.Dropdown):
    def __init__(
        self, data: int, controls: str = "default", id: Optional[Any] = None, **kwargs
    ):
        if id is None:
            id = self.generate_id(data, controls)
        options = get_args(DistributionPlot.PLOT_TYPE_OPTIONS)
        super().__init__(options, options[0], id=id, clearable=False, **kwargs)

    @classmethod
    def generate_id(cls, data: int, controls: str) -> Dict[str, Any]:
        return {"type": cls.__name__, "data": data, "controls": controls}


class ModelBarDropdown(dcc.Dropdown):
    def __init__(
        self, data: int, controls: str = "default", id: Optional[Any] = None, **kwargs
    ):
        if id is None:
            id = self.generate_id(data, controls)
        options = get_args(ModelBarPlot.GROUPING_OPTIONS)
        super().__init__(options, options[0], id=id, clearable=False, **kwargs)

    @classmethod
    def generate_id(cls, data: int, controls: str) -> Dict[str, Any]:
        return {"type": cls.__name__, "data": data, "controls": controls}


class EmbeddingPlot(dcc.Graph):
    def __init__(
        self, data: int, controls: str = "default", hover: str = "default", **kwargs
    ):
        super().__init__(
            id=self.generate_id(data, controls, hover), clear_on_unhover=True, **kwargs
        )

    @classmethod
    def generate_id(cls, data: int, controls: str, hover: str) -> Dict[str, Any]:
        return {
            "type": cls.__name__,
            "data": data,
            "controls": controls,
            "hover": hover,
            "hover_input": 1,
        }

    @classmethod
    def get_hover_index(cls, hover_data: Any) -> Optional[int]:
        return nested_get(hover_data, "points", 0, "customdata", 0)

    @classmethod
    def register_callbacks(cls, app: Dash, data: DataCache):
        @app.callback(
            Output(cls.generate_id(MATCH, MATCH, MATCH), "figure"),
            Input(JitterSlider.generate_id(MATCH, MATCH), "value"),
            Input(VariableDropdown.generate_id(MATCH, MATCH), "value"),
            Input(ClusterDropdown.generate_id(MATCH, MATCH), "value"),
            Input(HoverData.generate_id(MATCH, MATCH), "data"),
        )
        def callback(jitter, variable, cluster, hover):
            data_key = ctx.triggered_id["data"]
            df = data[data_key]
            dimensions = filter(lambda c: c[:2] == "Z_", df.columns)
            x = next(dimensions)
            y = next(dimensions)
            if cluster in df.columns:
                variable = cluster
            return cls.plot(df, x, y, variable, jitter, hover, seed=data_key)

    @staticmethod
    def plot(
        df: pd.DataFrame,
        x: str,
        y: str,
        variable: str,
        jitter: float = 0.0,
        hover: Optional[int] = None,
        seed: int = 42,
        fig_layout: Dict[str, Any] = DEFAULT_FIG_LAYOUT,
    ) -> Figure:
        def dfmod(var):
            df2 = pd.DataFrame(
                {x: df[x], y: df[y], var: df[var], "index": np.arange(df.shape[0])}
            )
            if jitter > 0:
                prng = np.random.default_rng(seed)
                df2[x] += prng.normal(0, jitter, df.shape[0])
                df2[y] += prng.normal(0, jitter, df.shape[0])
            return df2

        fig = None
        if is_categorical_dtype(df[variable]):
            df2 = dfmod(variable)
            fig = px.scatter(
                df2,
                x=x,
                y=y,
                color=variable,
                symbol=variable,
                category_orders={variable: df[variable].cat.categories},
                title="Embedding",
                custom_data=["index"],
            )
            fig.update_traces(hovertemplate=None, hoverinfo="none")
            ll = False
        else:
            ll = variable == "Local loss"
        if fig is None and ll and hover is not None:
            losses = [c for c in df.columns if c[:2] == "L_"]
            if len(losses) == df.shape[0]:
                lrange = (
                    df[losses].abs().min().quantile(0.05) * 0.9,
                    df[losses].abs().max().quantile(0.95) * 1.1,
                )
                variable = losses[hover]
                df2 = dfmod(variable)
                fig = px.scatter(
                    df2,
                    x=x,
                    y=y,
                    color=variable,
                    title=f"Alternative locations for item {df.index[hover]}",
                    opacity=0.8,
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
        if hover is not None:
            trace = px.scatter(df2.iloc[[hover]], x=x, y=y).update_traces(
                hoverinfo="skip",
                hovertemplate=None,
                marker=dict(
                    size=15, color="rgba(0,0,0,0)", line=dict(width=1, color="black")
                ),
            )
            fig.add_traces(trace.data)
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(**fig_layout, xaxis_title=None, yaxis_title=None)
        return fig


class ModelMatrixPlot(dcc.Graph):
    def __init__(
        self, data: int, controls: str = "default", hover: str = "default", **kwargs
    ):
        super().__init__(
            id=self.generate_id(data, controls, hover), clear_on_unhover=True, **kwargs
        )

    @classmethod
    def generate_id(cls, data: int, controls: str, hover: str) -> Dict[str, Any]:
        return {
            "type": cls.__name__,
            "data": data,
            "controls": controls,
            "hover": hover,
            "hover_input": 1,
        }

    @classmethod
    def get_hover_index(cls, hover_data: Any) -> Optional[int]:
        hover = nested_get(hover_data, "points", 0, "x")
        return int(hover) if hover is not None else None

    @classmethod
    def register_callbacks(cls, app: Dash, data: DataCache):
        @app.callback(
            Output(cls.generate_id(MATCH, MATCH, MATCH), "figure"),
            Input(HoverData.generate_id(MATCH, MATCH), "data"),
        )
        def callback(hover):
            data_key = ctx.triggered_id["data"]
            df = data[data_key]
            zs0 = next(filter(lambda c: c[:2] == "Z_", df.columns))
            bs = [c for c in df.columns if c[:2] == "B_"]
            return cls.plot(df, bs, zs0, hover)

    @staticmethod
    def plot(
        df: pd.DataFrame,
        coefficients: Sequence[str],
        sorting: Optional[str],
        hover: Optional[int] = None,
        fig_layout: Dict[str, Any] = DEFAULT_FIG_LAYOUT,
    ) -> Figure:
        if sorting is None:
            order_to_sorted = np.arange(df.shape[0])
        else:
            order_to_sorted = df[sorting].to_numpy().argsort()
        sorted_to_string = [str(i) for i in order_to_sorted]
        B_mat = df[coefficients].to_numpy()[order_to_sorted, :].T

        fig = px.imshow(
            B_mat,
            color_continuous_midpoint=0,
            aspect="auto",
            labels=dict(color="Coefficient", x="Data items sorted left to right"),
            title="Local models",
            color_continuous_scale="RdBu",
            y=coefficients,
            x=sorted_to_string,
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_layout(**fig_layout)
        fig.update_traces(hovertemplate="%{y} = %{z}<extra></extra>")
        if hover is not None:
            fig.add_vline(x=np.nonzero(order_to_sorted == hover)[0][0])
        return fig


class ModelBarPlot(dcc.Graph):
    def __init__(
        self, data: int, controls: str = "default", hover: str = "default", **kwargs
    ):
        super().__init__(id=self.generate_id(data, controls, hover), **kwargs)

    @classmethod
    def generate_id(cls, data: int, controls: str, hover: str) -> Dict[str, Any]:
        return {
            "type": cls.__name__,
            "data": data,
            "controls": controls,
            "hover": hover,
        }

    @classmethod
    def register_callbacks(cls, app: Dash, data: DataCache):
        @app.callback(
            Output(cls.generate_id(MATCH, MATCH, MATCH), "figure"),
            Input(ClusterDropdown.generate_id(MATCH, MATCH), "value"),
            Input(ModelBarDropdown.generate_id(MATCH, MATCH), "value"),
            Input(HoverData.generate_id(MATCH, MATCH), "data"),
        )
        def callback(cluster, grouping, hover):
            data_key = ctx.triggered_id["data"]
            df = data[data_key]
            coefficients = [c for c in df.columns if c[:2] == "B_"]
            return try_twice(
                lambda: cls.plot(df, coefficients, cluster, grouping, hover)
            )

    GROUPING_OPTIONS = Literal["Variables", "Clusters"]

    @staticmethod
    def plot(
        df: pd.DataFrame,
        coefficients: Sequence[str],
        cluster: Optional[str] = None,
        grouping: GROUPING_OPTIONS = "Variables",
        hover: Optional[int] = None,
        fig_layout: Dict[str, Any] = DEFAULT_FIG_LAYOUT,
    ) -> Figure:
        coefficient_range = df[coefficients].abs().quantile(0.95).max() * 1.1

        if hover is not None:
            fig = px.bar(
                pd.DataFrame(df[coefficients].iloc[hover]),
                range_y=(-coefficient_range, coefficient_range),
                color=coefficients,
                title=f"Local model for item {df.index[hover]}",
            )
            fig.update_layout(showlegend=False)
        elif cluster in df.columns and is_categorical_dtype(df[cluster]):
            if grouping == "Clusters":
                fig = px.bar(
                    df.groupby([cluster])[coefficients].mean().T,
                    color=cluster,
                    range_y=(-coefficient_range, coefficient_range),
                    title="Local models",
                    facet_col=cluster,
                )
                fig.update_annotations(visible=False)  # Remove facet labels
            else:
                fig = px.bar(
                    df.groupby([cluster])[coefficients].mean().T,
                    color=cluster,
                    range_y=(-coefficient_range, coefficient_range),
                    barmode="group",
                    title="Local models",
                )
        else:
            df2 = pd.DataFrame(df[coefficients].mean())
            fig = px.bar(
                df2,
                color=coefficients,
                range_y=(-coefficient_range, coefficient_range),
                title="Mean local model",
            )
            fig.update_layout(showlegend=False)
        fig.update_xaxes(title=None)
        fig.update_layout(**fig_layout, yaxis_title="Coefficient")
        fig.update_traces(hovertemplate="%{x} = %{y}<extra></extra>")
        return fig


class DistributionPlot(dcc.Graph):
    def __init__(
        self, data: int, controls: str = "default", hover: str = "default", **kwargs
    ):
        super().__init__(id=self.generate_id(data, controls, hover), **kwargs)

    @classmethod
    def generate_id(cls, data: int, controls: str, hover: str) -> Dict[str, Any]:
        return {
            "type": cls.__name__,
            "data": data,
            "controls": controls,
            "hover": hover,
        }

    @classmethod
    def register_callbacks(cls, app: Dash, data: DataCache):
        @app.callback(
            Output(cls.generate_id(MATCH, MATCH, MATCH), "figure"),
            Input(VariableDropdown.generate_id(MATCH, MATCH), "value"),
            Input(ClusterDropdown.generate_id(MATCH, MATCH), "value"),
            Input(HistogramDropdown.generate_id(MATCH, MATCH), "value"),
            Input(HoverData.generate_id(MATCH, MATCH), "data"),
        )
        def callback(variable, cluster, histogram, hover):
            data_key = ctx.triggered_id["data"]
            df = data[data_key]
            return try_twice(lambda: cls.plot(df, variable, histogram, cluster, hover))

    PLOT_TYPE_OPTIONS = Literal["Histogram", "Density"]

    def plot(
        df: pd.DataFrame,
        variable: str,
        plot_type: PLOT_TYPE_OPTIONS = "Histogram",
        cluster: Optional[str] = None,
        hover: Optional[int] = None,
        fig_layout: Dict[str, Any] = DEFAULT_FIG_LAYOUT,
    ) -> Figure:
        if cluster in df.columns and is_categorical_dtype(df[cluster]):
            if plot_type == "Histogram":
                fig = px.histogram(
                    df,
                    variable,
                    color=cluster,
                    title=f"{variable} histogram",
                    category_orders={cluster: df[cluster].cat.categories},
                )
            else:
                df2 = df.groupby(cluster)[variable]
                data = [df2.get_group(g) for g in df2.groups.keys()]
                clusters = [str(g) for g in df2.groups.keys()]
                colors = px.colors.qualitative.Plotly
                colors = colors * ((len(clusters) - 1) // len(colors) + 1)
                lengths = [len(i) > 1 for i in df2.groups.values()]
                if not all(lengths):
                    data = [d for d, l in zip(data, lengths) if l]
                    colors = [d for d, l in zip(colors, lengths) if l]
                    clusters = [d for d, l in zip(clusters, lengths) if l]
                fig = ff.create_distplot(data, clusters, show_hist=False, colors=colors)
                fig.layout.yaxis.domain = [0.31, 1]
                fig.layout.yaxis2.domain = [0, 0.29]
                fig.update_layout(
                    title=f"{variable} density plot",
                    legend=dict(title=cluster, traceorder="normal"),
                )
        else:
            if plot_type == "Histogram":
                fig = px.histogram(df, variable, title=f"{variable} histogram")
            else:
                fig = ff.create_distplot([df[variable]], [variable], show_hist=False)
                fig.layout.yaxis.domain = [0.21, 1]
                fig.layout.yaxis2.domain = [0, 0.19]
                fig.update_layout(showlegend=False, title=f"{variable} density plot")
        if hover is not None:
            fig.add_vline(x=df[variable].iloc[hover])
        fig.update_layout(**fig_layout, xaxis_title=None, yaxis_title=None)
        return fig


class HoverData(dcc.Store):
    def __init__(self, data: int, hover: str = "default", **kwargs):
        super().__init__(
            id=self.generate_id(data, hover), data=None, storage_type="memory", **kwargs
        )

    @classmethod
    def generate_id(cls, data: int, hover: str) -> Dict[str, Any]:
        return {"type": cls.__name__, "data": data, "hover": hover}

    @classmethod
    def register_callbacks(cls, app: Dash, data: Optional[DataCache] = None):
        input = EmbeddingPlot.generate_id(MATCH, ALL, MATCH)
        input["type"] = ALL

        @app.callback(
            Output(HoverData.generate_id(MATCH, MATCH), "data"),
            Input(input, "hoverData"),
            prevent_initial_call=True,
        )
        def hover_callback(inputs):
            tt = ctx.triggered_id["type"]
            if tt == "EmbeddingPlot":
                return first_not_none(inputs, EmbeddingPlot.get_hover_index)
            elif tt == "ModelMatrixPlot":
                return first_not_none(inputs, ModelMatrixPlot.get_hover_index)
            else:
                return first_not_none(inputs, nested_get, "points", 0, "customdata", 0)
            return None
