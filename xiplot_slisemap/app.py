"""
    Simple standalone Dash app.
"""
import sys
import os
from typing import Any, Optional

from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np

from load import slisemap_to_dataframe


def nested_get(obj: Any, *keys) -> Optional[Any]:
    for key in keys:
        if obj is None:
            return None
        obj = obj[key]
    return obj


def run_server(df: pd.DataFrame, debug=True):
    app = Dash(__name__)

    # Selection lists
    bs = [c for c in df.columns if c[:2] == "B_"]
    vars = ["Fidelity"] + [c for c in df.columns if c[0] == "X" or c[0] == "Y"]
    clusters = ["No clusters"] + [c for c in df.columns if c.startswith("Clusters")]

    # Jitter
    jitter_scale = 0.05
    jitters = {i: f"{i*jitter_scale:g}" for i in range(5)}
    jitters[0] = "No jitter"
    df["jitter_1"] = np.random.normal(0, jitter_scale, df.shape[0])
    df["jitter_2"] = np.random.normal(0, jitter_scale, df.shape[0])

    # B matrix
    # TODO the embedding->matrix hover is wrong
    order_to_sorted = df["Z_1"].argsort()
    index_to_sorted = np.argsort(order_to_sorted)
    sorted_to_index = np.argsort(index_to_sorted)
    B_mat = df[bs].to_numpy()[order_to_sorted, :].T

    # Coefficient extents:
    coefficient_range = df[bs].abs().quantile(0.95).max() * 1.1

    # Topbar
    topbar_style = {
        "display": "flex",
        "flex-direction": "row",
        "align-items": "center",
        "justify-content": "space-between",
        "gap": "0px",
    }
    header_style = {"flex-grow": "1", "flex-shrink": "1"}

    # Control knobs
    control_style_div = {
        "display": "flex",
        "flex-direction": "row",
        "align-items": "center",
        "justify-content": "right",
        "flex-wrap": "wrap",
        "gap": "0px",
        "flex-grow": "1",
        "flex-shrink": "1",
    }
    control_style = {"width": "15em"}
    control_jitter = dcc.Slider(
        id="jitter",
        min=min(jitters.keys()),
        max=max(jitters.keys()),
        marks=jitters,
        value=min(jitters.keys()),
    )
    control_jitter_wrapper = html.Div(
        children=[control_jitter],
        style={"display": "inline-block", **control_style},
    )
    control_var = dcc.Dropdown(
        vars, vars[0], id="variable", clearable=False, style=control_style
    )
    control_cluster = dcc.Dropdown(
        clusters, clusters[0], id="clusters", clearable=False, style=control_style
    )

    # Plots
    plot_style_div = {
        "display": "flex",
        "flex-direction": "row",
        "align-items": "stretch",
        "justify-content": "center",
        "align-content": "center",
        "flex-wrap": "wrap",
        "gap": "0px",
    }
    plot_style = {"min-width": "40em", "flex": "1 1 50%"}
    plot_embedding = dcc.Graph(id="embedding", style=plot_style, clear_on_unhover=True)
    plot_mat = dcc.Graph(id="matrix", style=plot_style, clear_on_unhover=True)
    plot_bar = dcc.Graph(id="barplot", style=plot_style)
    plot_hist = dcc.Graph(id="histogram", style=plot_style)
    hover_point = dcc.Store(id="hover_point", data=-1, storage_type="memory")

    # Layout
    app.layout = html.Div(
        children=[
            html.Div(
                children=[
                    html.H1(children="Interactive SLISEMAP", style=header_style),
                    html.Div(
                        children=[control_jitter_wrapper, control_var, control_cluster],
                        style=control_style_div,
                    ),
                ],
                style=topbar_style,
            ),
            html.Div(
                children=[plot_embedding, plot_mat, plot_bar, plot_hist],
                style=plot_style_div,
            ),
            hover_point,
        ]
    )

    # Callbacks
    @app.callback(
        Output(plot_embedding, "figure"),
        Input(control_jitter, "value"),
        Input(control_var, "value"),
        Input(control_cluster, "value"),
        Input(plot_embedding, "hoverData"),
    )
    def embedding_callback(jitter, variable, cluster, hover):
        # TODO bar on the left?
        if not cluster.startswith("No"):
            fig = px.scatter(
                df,
                x="Z_1",
                y="Z_2",
                color=cluster,
                symbol=cluster,
                category_orders={cluster: df[cluster].cat.categories},
                title="Embedding",
            )
        else:
            hover = nested_get(hover, "points", 0, "pointIndex")
            if hover is not None and hover > 0 and variable == "Fidelity":
                variable = f"L_{hover+1}"
                fig = px.scatter(
                    df,
                    x="Z_1",
                    y="Z_2",
                    color=variable,
                    title="Alternative Locations",
                    opacity=0.9,
                    color_continuous_scale="Viridis_r",
                    labels={variable: "Fidelity&nbsp;&nbsp;&nbsp;"},
                )
            else:
                fig = px.scatter(
                    df,
                    x="Z_1",
                    y="Z_2",
                    color=variable,
                    color_continuous_scale="Plasma_r",
                    title="Embedding",
                    opacity=0.9,
                    labels={variable: "Fidelity&nbsp;&nbsp;&nbsp;"}
                    if variable == "Fidelity"
                    else None,
                )

        if jitter > 0:
            fig.data[0].x += df["jitter_1"] * jitter
            fig.data[0].y += df["jitter_2"] * jitter
        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=20, autoexpand=True),
            xaxis_title=None,
            yaxis_title=None,
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        return fig

    @app.callback(
        Output(hover_point, "data"),
        Input(plot_embedding, "hoverData"),
        Input(plot_mat, "hoverData"),
    )
    def hover_callback(hover_embedding, hover_mat):
        hover = nested_get(hover_embedding, "points", 0, "pointIndex")
        if hover is not None:
            return hover
        hover = nested_get(hover_mat, "points", 0, "x")
        if hover is not None:
            return sorted_to_index[hover]
        return -1

    @app.callback(
        Output(plot_mat, "figure"),
        Input(hover_point, "data"),
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
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=20, autoexpand=True))
        if hover > -1:
            fig.add_vline(x=index_to_sorted[hover])
        return fig

    @app.callback(
        Output(plot_bar, "figure"),
        Input(control_cluster, "value"),
        Input(hover_point, "data"),
    )
    def barplot_callback(cluster, hover):
        if hover > -1:
            fig = px.bar(
                pd.DataFrame(df[bs].iloc[hover]),
                range_y=(-coefficient_range, coefficient_range),
                color=bs,
                title="Local model",
            )
            fig.update_layout(showlegend=False)
        elif not cluster.startswith("No"):
            fig = px.bar(
                df.groupby([cluster])[bs].mean().T,
                color=cluster,
                range_y=(-coefficient_range, coefficient_range),
                barmode="group",
                title="Local models",
            )
        else:
            df2 = pd.DataFrame(df[bs].mean())
            try:
                fig = px.bar(
                    df2,
                    color=bs,
                    range_y=(-coefficient_range, coefficient_range),
                    title="Mean local model",
                )
            except:
                fig = px.bar(
                    df2,
                    color=bs,
                    range_y=(-coefficient_range, coefficient_range),
                    title="Mean local model",
                )
            fig.update_layout(showlegend=False)
        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=20, autoexpand=True),
            xaxis_title=None,
            yaxis_title="Coefficient",
        )
        return fig

    @app.callback(
        Output(plot_hist, "figure"),
        Input(control_var, "value"),
        Input(control_cluster, "value"),
        Input(hover_point, "data"),
    )
    def histogram_callback(variable, cluster, hover):
        # TODO density plots?
        if not cluster.startswith("No"):
            fig = px.histogram(
                df,
                variable,
                color=cluster,
                title=f"{variable} histogram",
                category_orders={cluster: df[cluster].cat.categories},
            )
        else:
            try:
                fig = px.histogram(df, variable, title=f"{variable} histogram")
            except:
                fig = px.histogram(df, variable, title=f"{variable} histogram")
        if hover > -1:
            fig.add_vline(x=df[variable][hover])
        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=20, autoexpand=True),
            xaxis_title=None,
            yaxis_title="Count",
        )
        return fig

    # TODO better theme?
    # TODO sliemap column names
    app.scripts.config.serve_locally = True
    app.css.config.serve_locally = True
    app.run_server(debug=debug)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isdir(path):
            for path in [f for f in os.listdir(path) if f.endswith(".sm")]:
                print("Using:", path)
                break
        run_server(slisemap_to_dataframe(path, losses=True), debug=True)
    else:
        print(
            "Specify the path to a Slisemap object (or a directory containing a slisemap object) as a command line argument"
        )
