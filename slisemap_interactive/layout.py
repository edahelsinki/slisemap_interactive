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
    DataCache,
)


def register_callbacks(app: Dash, data: DataCache):
    HoverData.register_callbacks(app, data)
    EmbeddingPlot.register_callbacks(app, data)
    ModelMatrixPlot.register_callbacks(app, data)
    ModelBarPlot.register_callbacks(app, data)
    VariableHistogram.register_callbacks(app, data)


def setup_page(app: Dash, df: pd.DataFrame, data_key: int):
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
    hover_index = HoverData(data_key)
    controls = [
        JitterSlider(data_key, style={"display": "inline-block", **style_controls}),
        HistogramDropdown(data_key, style=style_controls),
        VariableDropdown(data_key, df, style=style_controls),
        ClusterDropdown(data_key, df, style=style_controls),
    ]
    plots = [
        EmbeddingPlot(data_key, style=style_plots),
        ModelMatrixPlot(data_key, style=style_plots),
        ModelBarPlot(data_key, style=style_plots),
        VariableHistogram(data_key, style=style_plots),
    ]

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
