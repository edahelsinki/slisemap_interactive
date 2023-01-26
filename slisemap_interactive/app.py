"""
    Simple standalone Dash app.
"""
import argparse
import sys
import os

from dash import Dash, html
import pandas as pd

from load import slisemap_to_dataframe
from plots import (
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
        prog="slisemap-dash",
        description="SLISEMAP - Dash:   A Dash app for interactively visualising SLISEMAP objects",
    )
    parser.add_argument(
        "PATH",
        help="The path to a Slisemap object (or a directory containing a Slisemap object)",
    )
    parser.add_argument("-d", "--debug", action="store_true", help="enable debug mode")
    parser.add_argument(
        "--no-debug", action="store_true", help="force disable debug mode"
    )
    args = parser.parse_args()
    path = args.PATH
    debug = not args.no_debug and args.debug
    if os.path.isdir(path):
        for path in [f for f in os.listdir(path) if f.endswith(".sm")]:
            print("Using:", path)
            break
    run_server(slisemap_to_dataframe(path, losses=True), debug)


def run_server(df: pd.DataFrame, debug=True):
    app = Dash(__name__)

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
                    html.H1(children="Interactive SLISEMAP", style=style_header),
                    html.Div(children=controls, style=style_control_div),
                ],
                style=style_topbar,
            ),
            html.Div(children=plots, style=style_plot_div),
            hover_index,
        ]
    )

    app.scripts.config.serve_locally = True
    app.css.config.serve_locally = True
    app.run(debug=debug)


if __name__ == "__main__":
    sys.argv.append("--debug")
    cli()
