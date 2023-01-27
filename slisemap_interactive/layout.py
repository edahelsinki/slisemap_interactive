"""
Create the layout and register callbacks
"""

from dash import Dash, html
import pandas as pd

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