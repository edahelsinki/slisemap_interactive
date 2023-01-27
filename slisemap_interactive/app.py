"""
    Simple standalone Dash app.
"""
import argparse
import sys
import os
from typing import Any, Dict

from dash import Dash, html
import pandas as pd

from slisemap_interactive.load import slisemap_to_dataframe
from slisemap_interactive.plots import (
    ClusterDropdown,
    EmbeddingPlot,
    HistogramDropdown,
    JitterSlider,
    ModelBarPlot,
    ModelMatrixPlot,
    VariableDropdown,
    HoverData,
    VariableHistogram,
)


def cli():
    parser = argparse.ArgumentParser(
        prog="slisemap-interactive",
        description="Slisemap - Interactive:   A Dash app for interactively visualising Slisemap objects",
    )
    parser.add_argument(
        "PATH",
        help="The path to a Slisemap object (or a directory containing a Slisemap object)",
    )
    parser.add_argument("--host", help="Host IP used to serve the application")
    parser.add_argument("-p", "--port", help="Port used to serve the application")
    parser.add_argument("-d", "--debug", action="store_true", help="enable debug mode")
    args = parser.parse_args()
    path = args.PATH
    if os.path.isdir(path):
        for path in [f for f in os.listdir(path) if f.endswith(".sm")]:
            print("Using:", path)
            break
    kwargs = {}
    if args.debug:
        kwargs["debug"] = True
    if args.host:
        kwargs["host"] = args.host
    if args.port:
        kwargs["port"] = args.port
    run_server(slisemap_to_dataframe(path, losses=True), **kwargs)


def run_server(df: pd.DataFrame, appargs: Dict[str, Any] = {}, **kwargs):
    appargs.setdefault("name", __name__)
    appargs.setdefault("serve_locally", True)
    appargs.setdefault("title", "Intercative Slisemap")
    app = Dash(**appargs)
    setup_page(app, df)
    kwargs.setdefault("debug", True)
    app.run(**kwargs)


def setup_page(app: Dash, df: pd.DataFrame):
    # Styles
    style_topbar = {
        "display": "flex",
        "flex-direction": "row",
        "align-items": "center",
        "justify-content": "space-between",
        "gap": "0px",
    }
    style_header = {"flex-grow": "1", "flex-shrink": "1"}
    style_control_div = {
        "display": "flex",
        "flex-direction": "row",
        "align-items": "center",
        "justify-content": "right",
        "flex-wrap": "wrap",
        "gap": "0px",
        "flex-grow": "1",
        "flex-shrink": "1",
    }
    style_controls = {"width": "15em"}
    style_plot_div = {
        "display": "flex",
        "flex-direction": "row",
        "align-items": "stretch",
        "justify-content": "center",
        "align-content": "center",
        "flex-wrap": "wrap",
        "gap": "0px",
    }
    style_plots = {"min-width": "40em", "flex": "1 1 50%"}

    # Elements
    hover_index = HoverData()
    controls = [
        JitterSlider(style={"display": "inline-block", **style_controls}),
        HistogramDropdown(style=style_controls),
        VariableDropdown(df, style=style_controls),
        ClusterDropdown(df, style=style_controls),
    ]
    plots = [
        EmbeddingPlot(style=style_plots),
        ModelMatrixPlot(style=style_plots),
        ModelBarPlot(style=style_plots),
        VariableHistogram(style=style_plots),
    ]

    # Register callbacks
    HoverData.register_callbacks(app)
    EmbeddingPlot.register_callbacks(app, df)
    ModelMatrixPlot.register_callbacks(app, df)
    ModelBarPlot.register_callbacks(app, df)
    VariableHistogram.register_callbacks(app, df)

    # Layout
    app.layout = html.Div(
        children=[
            html.Div(
                children=[
                    html.H1(children="Interactive Slisemap", style=style_header),
                    html.Div(children=controls, style=style_control_div),
                ],
                style=style_topbar,
            ),
            html.Div(children=plots, style=style_plot_div),
            hover_index,
        ]
    )


if __name__ == "__main__":
    sys.argv.append("--debug")
    cli()
