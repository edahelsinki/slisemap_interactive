"""
    Simple standalone Dash app.
"""
import sys
import os

from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np

from load import slisemap_to_dataframe


def run_server(df: pd.DataFrame, debug=True):
    app = Dash(__name__)

    # Config
    jitter_scale = 0.025

    # Selection lists
    bs = [c for c in df.columns if c[:2] == "B_"]
    vars = ["Fidelity"] + [c for c in df.columns if c[0] == "X" or c[0] == "Y"]
    clusters = ["No clusters"] + [c for c in df.columns if c.startswith("Clusters")]
    jitters = {i: f"{i*jitter_scale:g}" for i in range(5)}
    jitters[0] = "No jitter"

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
        min=min(jitters.keys()),
        max=max(jitters.keys()),
        marks=jitters,
        value=min(jitters.keys()),
    )
    control_jitter_wrapper = html.Div(
        children=[control_jitter],
        style={"display": "inline-block", **control_style},
    )
    control_var = dcc.Dropdown(vars, vars[0], clearable=False, style=control_style)
    control_cluster = dcc.Dropdown(
        clusters, clusters[0], clearable=False, style=control_style
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
    plot_embedding = dcc.Graph(style=plot_style)
    plot_mat = dcc.Graph(style=plot_style)
    plot_bar = dcc.Graph(style=plot_style)
    plot_hist = dcc.Graph(style=plot_style)

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
        ]
    )

    # Callbacks
    @app.callback(
        Output(plot_embedding, "figure"),
        Input(control_jitter, "value"),
        Input(control_var, "value"),
        Input(control_cluster, "value"),
    )
    def embedding_callback(jitter, variable, cluster):
        # TODO handle jitter
        if not cluster.startswith("No"):
            fig = px.scatter(
                df, x="Z_1", y="Z_2", color=cluster, symbol=cluster, title="Embedding"
            )
        else:
            fig = px.scatter(df, x="Z_1", y="Z_2", color=variable, title="Embedding")
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=20, autoexpand=True))
        return fig

    @app.callback(
        Output(plot_mat, "figure"),
        Input(control_cluster, "value"),
    )
    def matplot_callback(cluster):
        fig = px.imshow(
            df[bs].T,
            color_continuous_midpoint=0,
            aspect="auto",
            labels=dict(color="Coefficient"),
            title="Local models",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=20, autoexpand=True))
        return fig

    @app.callback(
        Output(plot_bar, "figure"),
        Input(control_cluster, "value"),
    )
    def barplot_callback(cluster):
        if not cluster.startswith("No"):
            fig = px.bar(
                df.groupby([cluster]).mean(numeric_only=True).reset_index(),
                # TODO pivot wide to long
                # color=cluster,
                x=cluster,
                y=bs,
                barmode="group",
                title="Local models",
            )
        else:
            # TODO fix table
            fig = px.bar(df[bs].mean(numeric_only=True), y=bs, title="Mean model")
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=20, autoexpand=True))
        return fig

    @app.callback(
        Output(plot_hist, "figure"),
        Input(control_var, "value"),
        Input(control_cluster, "value"),
    )
    def histogram_callback(variable, cluster):
        if not cluster.startswith("No"):
            fig = px.histogram(
                df, variable, color=cluster, title=f"{variable} histogram"
            )
        else:
            fig = px.histogram(df, variable, title=f"{variable} histogram")
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=20, autoexpand=True))
        return fig

    # TODO handle hover
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
        run_server(slisemap_to_dataframe(path), debug=True)
    else:
        print(
            "Specify the path to a Slisemap object (or a directory containing a slisemap object) as a command line argument"
        )
