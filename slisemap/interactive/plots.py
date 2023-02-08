"""
Functions for generating dynamic plots.
"""

from typing import Any, Dict, Optional

from dash import Dash, html, dcc, Input, Output, ctx, MATCH, ALL
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import numpy as np


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
        **kwargs,
    ):
        values = np.linspace(0.0, scale, steps)
        marks = {0: "No jitter"}
        for v in values[1:]:
            marks[v] = f"{v:g}"
        id = self.generate_id(data, controls)
        slider = dcc.Slider(id=id, min=0.0, max=scale, marks=marks, value=0.0)
        super().__init__(children=[slider], **kwargs)

    @classmethod
    def generate_id(cls, data: int, controls: str) -> Dict[str, Any]:
        return {"type": cls.__name__, "data": data, "controls": controls}


class VariableDropdown(dcc.Dropdown):
    def __init__(
        self, data: int, df: pd.DataFrame, controls: str = "default", **kwargs
    ):
        vars = ["Local loss"] + [c for c in df.columns if c[0] in ("X", "Y", "B")]
        id = self.generate_id(data, controls)
        super().__init__(vars, vars[0], id=id, clearable=False, **kwargs)

    @classmethod
    def generate_id(cls, data: int, controls: str) -> Dict[str, Any]:
        return {"type": cls.__name__, "data": data, "controls": controls}


class ClusterDropdown(dcc.Dropdown):
    def __init__(
        self, data: int, df: pd.DataFrame, controls: str = "default", **kwargs
    ):
        clusters = ["No clusters"] + [c for c in df.columns if c.startswith("Clusters")]
        id = self.generate_id(data, controls)
        super().__init__(clusters, clusters[0], id=id, clearable=False, **kwargs)

    @classmethod
    def generate_id(cls, data: int, controls: str) -> Dict[str, Any]:
        return {"type": cls.__name__, "data": data, "controls": controls}


class HistogramDropdown(dcc.Dropdown):
    def __init__(self, data: int, controls: str = "default", **kwargs):
        id = self.generate_id(data, controls)
        super().__init__(
            ["Histogram", "Density"], "Histogram", id=id, clearable=False, **kwargs
        )

    @classmethod
    def generate_id(cls, data: int, controls: str) -> Dict[str, Any]:
        return {"type": cls.__name__, "data": data, "controls": controls}


class ModelBarDropdown(dcc.Dropdown):
    def __init__(self, data: int, controls: str = "default", **kwargs):
        id = self.generate_id(data, controls)
        super().__init__(
            ["Variables", "Clusters"], "Variables", id=id, clearable=False, **kwargs
        )

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
    def get_hover_index(cls, data: DataCache, hover_data: Any):
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
        def embedding_callback(jitter, variable, cluster, hover):
            data_key = ctx.triggered_id["data"]
            df = data[data_key]
            dimensions = filter(lambda c: c[:2] == "Z_", df.columns)
            dimx = next(dimensions)
            dimy = next(dimensions)

            def dfmod(var):
                df2 = pd.DataFrame(
                    {
                        dimx: df[dimx],
                        dimy: df[dimy],
                        var: df[var],
                        "index": np.arange(df.shape[0]),
                    }
                )
                if jitter > 0:
                    prng = np.random.default_rng(data_key)
                    df2[dimx] += prng.normal(0, jitter, df.shape[0])
                    df2[dimy] += prng.normal(0, jitter, df.shape[0])
                return df2

            if not cluster.startswith("No"):
                variable = cluster
                df2 = dfmod(cluster)
                fig = px.scatter(
                    df2,
                    x=dimx,
                    y=dimy,
                    color=cluster,
                    symbol=cluster,
                    category_orders={cluster: df[cluster].cat.categories},
                    title="Embedding",
                    custom_data=["index"],
                )
                fig.update_traces(hovertemplate=None, hoverinfo="none")
            else:
                losses = [c for c in df.columns if c[:2] == "L_"]
                ll = variable == "Local loss"
                if hover is not None and ll and len(losses) > 0:
                    lrange = (
                        df[losses].abs().min().quantile(0.05) * 0.9,
                        df[losses].abs().max().quantile(0.95) * 1.1,
                    )
                    variable = losses[hover]
                    df2 = dfmod(variable)
                    fig = px.scatter(
                        df2,
                        x=dimx,
                        y=dimy,
                        color=variable,
                        title=f"Alternative locations for item {df.index[hover]}",
                        opacity=0.8,
                        color_continuous_scale="Viridis_r",
                        labels={variable: "Local loss&nbsp;"},
                        custom_data=["index"],
                        range_color=lrange,
                    )
                else:
                    df2 = dfmod(variable)
                    fig = px.scatter(
                        df2,
                        x=dimx,
                        y=dimy,
                        color=variable,
                        color_continuous_scale="Plasma_r",
                        title="Embedding",
                        opacity=0.8,
                        labels={"Local loss": "Local loss&nbsp;"} if ll else None,
                        custom_data=["index"],
                    )
                if ll:
                    fig.update_traces(hovertemplate=None, hoverinfo="none")
            if hover is not None:
                trace = px.scatter(df2.iloc[[hover]], x=dimx, y=dimy).update_traces(
                    hoverinfo="skip",
                    hovertemplate=None,
                    marker=dict(
                        size=15,
                        color="rgba(0,0,0,0)",
                        line=dict(width=1, color="black"),
                    ),
                )
                fig.add_traces(trace.data)
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            fig.update_layout(
                margin=dict(l=10, r=10, t=30, b=20, autoexpand=True),
                xaxis_title=None,
                yaxis_title=None,
                template="plotly_white",
                uirevision=True,
            )
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
    def get_hover_index(cls, data: DataCache, hover_data: Any):
        return nested_get(hover_data, "points", 0, "x")

    @classmethod
    def register_callbacks(cls, app: Dash, data: DataCache):
        @app.callback(
            Output(cls.generate_id(MATCH, MATCH, MATCH), "figure"),
            Input(HoverData.generate_id(MATCH, MATCH), "data"),
        )
        def matplot_callback(hover):
            data_key = ctx.triggered_id["data"]
            df = data[data_key]
            zs0 = next(filter(lambda c: c[:2] == "Z_", df.columns))
            bs = [c for c in df.columns if c[:2] == "B_"]
            order_to_sorted = df[zs0].argsort()
            sorted_to_string = [str(i) for i in order_to_sorted]
            B_mat = df[bs].to_numpy()[order_to_sorted, :].T

            fig = px.imshow(
                B_mat,
                color_continuous_midpoint=0,
                aspect="auto",
                labels=dict(color="Coefficient"),
                title="Local models",
                color_continuous_scale="RdBu",
                y=bs,
                x=sorted_to_string,
            )
            fig.update_xaxes(showticklabels=False)
            fig.update_layout(
                margin=dict(l=10, r=10, t=30, b=20, autoexpand=True),
                template="plotly_white",
                uirevision=True,
            )
            fig.update_traces(hovertemplate="%{y} = %{z}<extra></extra>")
            if hover is not None:
                fig.add_vline(x=np.where(order_to_sorted == hover)[0][0])
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
        def barplot_callback(cluster, grouping, hover):
            data_key = ctx.triggered_id["data"]
            df = data[data_key]
            coefficients = [c for c in df.columns if c[:2] == "B_"]
            coefficient_range = df[coefficients].abs().quantile(0.95).max() * 1.1

            if hover is not None:
                fig = px.bar(
                    pd.DataFrame(df[coefficients].iloc[hover]),
                    range_y=(-coefficient_range, coefficient_range),
                    color=coefficients,
                    title=f"Local model for item {df.index[hover]}",
                )
                fig.update_layout(showlegend=False)
            elif not cluster.startswith("No"):
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
                try:
                    fig = px.bar(
                        df2,
                        color=coefficients,
                        range_y=(-coefficient_range, coefficient_range),
                        title="Mean local model",
                    )
                except:
                    fig = px.bar(
                        df2,
                        color=coefficients,
                        range_y=(-coefficient_range, coefficient_range),
                        title="Mean local model",
                    )
                fig.update_layout(showlegend=False)
            fig.update_xaxes(title=None)
            fig.update_layout(
                margin=dict(l=10, r=10, t=30, b=20, autoexpand=True),
                yaxis_title="Coefficient",
                template="plotly_white",
                uirevision=True,
            )
            fig.update_traces(hovertemplate="%{x} = %{y}<extra></extra>")
            return fig


class VariableHistogram(dcc.Graph):
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
        def histogram_callback(variable, cluster, histogram, hover):
            data_key = ctx.triggered_id["data"]
            df = data[data_key]

            if not cluster.startswith("No"):
                if histogram == "Histogram":
                    fig = px.histogram(
                        df,
                        variable,
                        color=cluster,
                        title=f"{variable} histogram",
                        category_orders={cluster: df[cluster].cat.categories},
                    )
                else:
                    df2 = df.groupby(cluster)[variable]
                    fig = ff.create_distplot(
                        [df2.get_group(g) for g in df[cluster].cat.categories],
                        [str(g) for g in df[cluster].cat.categories],
                        show_hist=False,
                        colors=px.colors.qualitative.Plotly,
                    )
                    fig.layout.yaxis.domain = [0.31, 1]
                    fig.layout.yaxis2.domain = [0, 0.29]
                    fig.update_layout(
                        title=f"{variable} density plot",
                        legend=dict(title="Clusters", traceorder="normal"),
                    )
            else:
                if histogram == "Histogram":
                    try:
                        fig = px.histogram(df, variable, title=f"{variable} histogram")
                    except ValueError:  # Sometimes the plot fails on page load
                        fig = px.histogram(df, variable, title=f"{variable} histogram")
                else:
                    fig = ff.create_distplot(
                        [df[variable]], [variable], show_hist=False
                    )
                    fig.layout.yaxis.domain = [0.21, 1]
                    fig.layout.yaxis2.domain = [0, 0.19]
                    fig.update_layout(
                        showlegend=False, title=f"{variable} density plot"
                    )
            if hover is not None:
                fig.add_vline(x=df[variable].iloc[hover])
            fig.update_layout(
                margin=dict(l=10, r=10, t=30, b=20, autoexpand=True),
                xaxis_title=None,
                yaxis_title=None,
                template="plotly_white",
                uirevision=True,
            )
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
                for input in inputs:
                    hover = EmbeddingPlot.get_hover_index(data, input)
                    if hover is not None:
                        return hover
            elif tt == "ModelMatrixPlot":
                for input in inputs:
                    hover = ModelMatrixPlot.get_hover_index(data, input)
                    if hover is not None:
                        return int(hover)
            else:
                for input in inputs:
                    hover = nested_get(input, "points", 0, "customdata", 0)
                    if hover is not None:
                        return hover
            return None
