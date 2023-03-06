"""
Create the layout and register callbacks
"""

from dash import Dash, html
import pandas as pd

from slisemap_interactive.plots import (
    BarGroupingDropdown,
    ClusterDropdown,
    EmbeddingPlot,
    DensityTypeDropdown,
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
        "flexDirection": "row",
        "alignItems": "center",
        "justifyContent": "right",
        "flexWrap": "wrap",
        "gap": "0px",
        "padding": "0.4rem",
        "background": "#FDF3FF",
        "borderRadius": "4px",
        "border": "thin solid #D0D0DF",
        "boxShadow": "0px 2px 3px 0px hsla(0, 0%, 0%, 0.14)",
        "marginBottom": "0.5rem",
    }
    style_header = {
        "flex": "1 1",
        "paddingLeft": "0.3rem",
        "paddingRight": "0.3rem",
        "marginTop": "0px",
        "marginBottom": "0px",
    }
    style_controls = {"width": "14em"}
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
        VariableDropdown(df, data_key, style=style_controls),
        ClusterDropdown(df, data_key, style=style_controls),
        JitterSlider(data_key, style={"display": "inline-block", **style_controls}),
        BarGroupingDropdown(data_key, style=style_controls),
        DensityTypeDropdown(data_key, style=style_controls),
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
