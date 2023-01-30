"""
Functions for generating dynamic plots.
"""

from typing import Any, Optional

from dash import Dash, html, dcc, Input, Output, ctx, MATCH, ALL
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import numpy as np


def nested_get(obj: Any, *keys) -> Optional[Any]:
    for key in keys:
        if obj is None or len(obj) == 0:
            return None
        obj = obj[key]
    return obj


class JitterSlider(html.Div):
    def __init__(
        self, scale: float = 0.2, steps: int = 5, controls: str = "default", **kwargs
    ):
        values = np.linspace(0.0, scale, steps)
        marks = {0: "No jitter"}
        for v in values[1:]:
            marks[v] = f"{v:g}"
        id = {"type": "JitterSlider", "controls": controls}
        slider = dcc.Slider(id=id, min=0.0, max=scale, marks=marks, value=0.0)
        super().__init__(children=[slider], **kwargs)


class VariableDropdown(dcc.Dropdown):
    def __init__(self, df: pd.DataFrame, controls: str = "default", **kwargs):
        vars = ["Local loss"] + [c for c in df.columns if c[0] in ("X", "Y", "B")]
        id = {"type": "VariableDropdown", "controls": controls}
        super().__init__(vars, vars[0], id=id, clearable=False, **kwargs)


class ClusterDropdown(dcc.Dropdown):
    def __init__(self, df: pd.DataFrame, controls: str = "default", **kwargs):
        clusters = ["No clusters"] + [c for c in df.columns if c.startswith("Clusters")]
        id = {"type": "ClusterDropdown", "controls": controls}
        super().__init__(clusters, clusters[0], id=id, clearable=False, **kwargs)


class HistogramDropdown(dcc.Dropdown):
    def __init__(self, controls: str = "default", **kwargs):
        new_var = {"type": "HistogramDropdown", "controls": controls}
        id = new_var
        super().__init__(
            ["Histogram", "Density"], "Histogram", id=id, clearable=False, **kwargs
        )


class EmbeddingPlot(dcc.Graph):
    def __init__(self, controls: str = "default", hover: str = "default", **kwargs):
        id = {
            "type": "EmbeddingPlot",
            "controls": controls,
            "hover": hover,
            "hoverInput": 1,
        }
        super().__init__(id=id, clear_on_unhover=True, **kwargs)

    @classmethod
    def register_callbacks(cls, app: Dash, df: pd.DataFrame):
        dimensions = [c for c in df.columns if c[:2] == "Z_"]
        losses = [c for c in df.columns if c[:2] == "L_"]
        jitter_x = np.random.normal(0, 1, df.shape[0])
        jitter_y = np.random.normal(0, 1, df.shape[0])

        @app.callback(
            Output(
                {
                    "type": "EmbeddingPlot",
                    "controls": MATCH,
                    "hover": MATCH,
                    "hoverInput": 1,
                },
                "figure",
            ),
            Input({"type": "JitterSlider", "controls": MATCH}, "value"),
            Input({"type": "VariableDropdown", "controls": MATCH}, "value"),
            Input({"type": "ClusterDropdown", "controls": MATCH}, "value"),
            Input({"type": "HoverData", "hover": MATCH}, "data"),
        )
        def embedding_callback(jitter, variable, cluster, hover):
            def dfmod(var):
                df2 = pd.DataFrame(
                    {
                        dimensions[0]: df[dimensions[0]],
                        dimensions[1]: df[dimensions[1]],
                        var: df[var],
                        "index": np.arange(df.shape[0]),
                    }
                )
                if jitter > 0:
                    df2[dimensions[0]] += jitter_x * jitter
                    df2[dimensions[1]] += jitter_y * jitter
                return df2

            if not cluster.startswith("No"):
                variable = cluster
                df2 = dfmod(cluster)
                fig = px.scatter(
                    df2,
                    x=dimensions[0],
                    y=dimensions[1],
                    color=cluster,
                    symbol=cluster,
                    category_orders={cluster: df[cluster].cat.categories},
                    title="Embedding",
                    custom_data=["index"],
                )
                fig.update_traces(hovertemplate=None, hoverinfo="none")
            else:
                ll = variable == "Local loss"
                if hover is not None and ll and len(losses) > 0:
                    variable = losses[hover]
                    df2 = dfmod(variable)
                    fig = px.scatter(
                        df2,
                        x=dimensions[0],
                        y=dimensions[1],
                        color=variable,
                        title="Alternative Locations",
                        opacity=0.8,
                        color_continuous_scale="Viridis_r",
                        labels={variable: "Local loss&nbsp;"},
                        custom_data=["index"],
                    )
                else:
                    df2 = dfmod(variable)
                    fig = px.scatter(
                        df2,
                        x=dimensions[0],
                        y=dimensions[1],
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
                trace = px.scatter(
                    df2.iloc[[hover]], x=dimensions[0], y=dimensions[1]
                ).update_traces(
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
    def __init__(self, controls: str = "default", hover: str = "default", **kwargs):
        id = {
            "type": "ModelMatrixPlot",
            "controls": controls,
            "hover": hover,
            "hoverInput": 1,
        }
        super().__init__(id=id, clear_on_unhover=True, **kwargs)

    @classmethod
    def register_callbacks(cls, app, df):
        for c in df.columns:
            if c[:2] == "Z_":
                zs0 = c
                break
        bs = [c for c in df.columns if c[:2] == "B_"]
        order_to_sorted = df[zs0].argsort()
        index_to_sorted = np.argsort(order_to_sorted)
        sorted_to_string = [str(i) for i in np.arange(df.shape[0])[order_to_sorted]]

        B_mat = df[bs].to_numpy()[order_to_sorted, :].T

        @app.callback(
            Output(
                {
                    "type": "ModelMatrixPlot",
                    "controls": MATCH,
                    "hover": MATCH,
                    "hoverInput": 1,
                },
                "figure",
            ),
            Input({"type": "HoverData", "hover": MATCH}, "data"),
        )
        def matplot_callback(hover):
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
            # fig.data[0]["x"] = [f"{i}" for i in sorted_to_index]
            fig.update_xaxes(showticklabels=False)
            fig.update_layout(
                margin=dict(l=10, r=10, t=30, b=20, autoexpand=True),
                template="plotly_white",
                uirevision=True,
            )
            fig.update_traces(
                # customdata=sorted_to_index.to_numpy(),
                hovertemplate="%{y} = %{z}<br>index = %{x}<extra></extra>",  # "%{y} = %{z}<br>index = %{customdata}<extra></extra>",
            )
            if hover is not None:
                fig.add_vline(x=index_to_sorted[hover])
            return fig


class ModelBarPlot(dcc.Graph):
    def __init__(self, controls: str = "default", hover: str = "default", **kwargs):
        id = {"type": "ModelBarPlot", "controls": controls, "hover": hover}
        super().__init__(id=id, **kwargs)

    @classmethod
    def register_callbacks(cls, app, df):
        coefficients = [c for c in df.columns if c[:2] == "B_"]
        coefficient_range = df[coefficients].abs().quantile(0.95).max() * 1.1

        @app.callback(
            Output(
                {"type": "ModelBarPlot", "controls": MATCH, "hover": MATCH}, "figure"
            ),
            Input({"type": "ClusterDropdown", "controls": MATCH}, "value"),
            Input({"type": "HoverData", "hover": MATCH}, "data"),
        )
        def barplot_callback(cluster, hover):
            if hover is not None:
                fig = px.bar(
                    pd.DataFrame(df[coefficients].iloc[hover]),
                    range_y=(-coefficient_range, coefficient_range),
                    color=coefficients,
                    title="Local model",
                )
                fig.update_layout(showlegend=False)
            elif not cluster.startswith("No"):
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
            fig.update_layout(
                margin=dict(l=10, r=10, t=30, b=20, autoexpand=True),
                xaxis_title=None,
                yaxis_title="Coefficient",
                template="plotly_white",
                uirevision=True,
            )
            return fig


class VariableHistogram(dcc.Graph):
    def __init__(self, controls: str = "default", hover: str = "default", **kwargs):
        id = {"type": "VariableHistogram", "controls": controls, "hover": hover}
        super().__init__(id=id, **kwargs)

    @classmethod
    def register_callbacks(cls, app, df):
        @app.callback(
            Output(
                {"type": "VariableHistogram", "controls": MATCH, "hover": MATCH},
                "figure",
            ),
            Input({"type": "VariableDropdown", "controls": MATCH}, "value"),
            Input({"type": "ClusterDropdown", "controls": MATCH}, "value"),
            Input({"type": "HistogramDropdown", "controls": MATCH}, "value"),
            Input({"type": "HoverData", "hover": MATCH}, "data"),
        )
        def histogram_callback(variable, cluster, histogram, hover):
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
                fig.add_vline(x=df[variable][hover])
            fig.update_layout(
                margin=dict(l=10, r=10, t=30, b=20, autoexpand=True),
                xaxis_title=None,
                yaxis_title=None,
                template="plotly_white",
                uirevision=True,
            )
            return fig


class HoverData(dcc.Store):
    def __init__(self, hover: str = "default", **kwargs):
        id = {"type": "HoverData", "hover": hover}
        super().__init__(id=id, data=None, storage_type="memory", **kwargs)

    @classmethod
    def register_callbacks(cls, app):
        @app.callback(
            Output({"type": "HoverData", "hover": MATCH}, "data"),
            Input(
                {"type": ALL, "controls": ALL, "hover": MATCH, "hoverInput": 1},
                "hoverData",
            ),
            prevent_initial_call=True,
        )
        def hover_callback(inputs):
            tt = ctx.triggered_id["type"]
            if tt == "EmbeddingPlot":
                for input in inputs:
                    hover = nested_get(input, "points", 0, "customdata", 0)
                    if hover is not None:
                        return hover
            elif tt == "ModelMatrixPlot":
                for input in inputs:
                    hover = nested_get(input, "points", 0, "x")
                    if hover is not None:
                        return int(hover)
            return None
