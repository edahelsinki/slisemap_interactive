"""
Create the layout and register callbacks
"""

from dash import Dash, html
import pandas as pd

from slisemap_interactive.plots import (
    ModelBarDropdown,
    ClusterDropdown,
    EmbeddingPlot,
    HistogramDropdown,
    JitterSlider,
    ModelBarPlot,
    ModelMatrixPlot,
    VariableDropdown,
    HoverData,
    DistributionPlot,
    DataCache,
)


def register_callbacks(app: Dash, data: DataCache):
    """Register callbacks for updating the plots.

    Args:
        app: Dash app.
        data: Dictionary to get the dataframes from.
    """
    HoverData.register_callbacks(app, data)
    EmbeddingPlot.register_callbacks(app, data)
    ModelMatrixPlot.register_callbacks(app, data)
    ModelBarPlot.register_callbacks(app, data)
    DistributionPlot.register_callbacks(app, data)


def page_with_all_plots(df: pd.DataFrame, data_key: int) -> html.Div:
    """Generate the layout for a webpage with all plots.

    Args:
        df: Current dataframe (for dropdowns etc.).
        data_key: Key to the dataframe (for updating the plots etc.).

    Returns:
        A div containing the page (set `app.layout = setup_page(...)`).
    """
    # Styles
    style_topbar = {
        "display": "flex",
        "flex-direction": "row",
        "align-items": "center",
        "justify-content": "right",
        "flex-wrap": "wrap",
        "gap": "0px",
        "padding": "0.2em",
        "border": "thin solid lightgrey",
        "margin-bottom": "0.4em",
    }
    style_header = {
        "flex-grow": "1",
        "flex-shrink": "1",
        "margin-top": "0px",
        "margin-bottom": "0px",
    }
    style_controls = {"width": "14em"}
    style_plot_area = {
        "display": "flex",
        "flex-direction": "row",
        "align-items": "stretch",
        "justify-content": "center",
        "align-content": "center",
        "flex-wrap": "wrap",
        "gap": "0px",
    }
    style_plot = {"min-width": "40em", "flex": "1 1 50%"}

    # Elements
    hover_index = HoverData(data_key)
    topbar = [
        html.H1(children="Interactive Slisemap", style=style_header),
        VariableDropdown(data_key, df, style=style_controls),
        ClusterDropdown(data_key, df, style=style_controls),
        JitterSlider(data_key, style={"display": "inline-block", **style_controls}),
        ModelBarDropdown(data_key, style=style_controls),
        HistogramDropdown(data_key, style=style_controls),
    ]
    plots = [
        EmbeddingPlot(data_key, style=style_plot),
        ModelMatrixPlot(data_key, style=style_plot),
        ModelBarPlot(data_key, style=style_plot),
        DistributionPlot(data_key, style=style_plot),
    ]

    # Layout
    return html.Div(
        children=[
            html.Div(children=topbar, style=style_topbar),
            html.Div(children=plots, style=style_plot_area),
            hover_index,
        ]
    )
