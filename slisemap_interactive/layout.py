"""Create the layout and register callbacks."""

import pandas as pd
from dash import Dash, html

from slisemap_interactive.plots import (
    BarGroupingDropdown,
    ClusterDropdown,
    ContourCheckbox,
    DataCache,
    DensityTypeDropdown,
    DistributionPlot,
    EmbeddingPlot,
    HoverData,
    JitterSlider,
    ModelBarPlot,
    ModelMatrixPlot,
    VariableDropdown,
)


def register_callbacks(app: Dash, data: DataCache) -> None:
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
        "flexDirection": "row",
        "alignItems": "center",
        "justifyContent": "right",
        "flexWrap": "wrap",
        "padding": "0.4rem",
        "background": "#FDF3FF",
        "borderRadius": "4px",
        "border": "thin solid #D0D0DF",
        "boxShadow": "0px 2px 3px 0px hsla(0, 0%, 0%, 0.14)",
        "marginBottom": "0.5rem",
        "gap": "0px 0.2rem",
    }
    style_header = {
        "paddingLeft": "0.3rem",
        "paddingRight": "0.3rem",
        "marginTop": "0px",
        "marginBottom": "0px",
    }
    style_plot_area = {
        "display": "flex",
        "flexDirection": "row",
        "alignItems": "stretch",
        "justifyContent": "center",
        "alignContent": "center",
        "flexWrap": "wrap",
        "gap": "0px",
    }
    style_plot = {"minWidth": "35rem", "flex": "1 1 50%"}

    # Elements
    hover_index = HoverData(data_key)
    topbar = [
        html.H1(children="Interactive Slisemap", style=style_header),
        html.Div(style={"flex": "1"}),
        VariableDropdown(df, data_key, style={"minWidth": "14em"}),
        ClusterDropdown(df, data_key, style={"minWidth": "14em"}),
        ContourCheckbox(data_key, style={"paddingRight": "0.3rem"}),
        JitterSlider(data_key, style={"display": "inline-block", "minWidth": "14em"}),
        BarGroupingDropdown(data_key, style={"minWidth": "10em"}),
        DensityTypeDropdown(data_key, style={"minWidth": "10em"}),
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
