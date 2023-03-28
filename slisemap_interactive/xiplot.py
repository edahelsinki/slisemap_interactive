"""
    Hooks for connecting to xiplot (using entry points in 'pyproject.toml').
"""
from typing import Any, Callable, Dict, List

import pandas as pd
from dash import ALL, MATCH, Input, Output, State, dcc, html
from xiplot.plugin import (
    STORE_CLICKED_ID,
    STORE_DATAFRAME_ID,
    STORE_HOVERED_ID,
    APlot,
    FlexRow,
    PdfButton,
    PlotData,
)

from slisemap_interactive.load import slisemap_to_dataframe
from slisemap_interactive.plots import (
    BarGroupingDropdown,
    ClusterDropdown,
    DistributionPlot,
    EmbeddingPlot,
    JitterSlider,
    LinearImpact,
    ModelBarPlot,
    ModelMatrixPlot,
    PredictionDropdown,
    VariableDropdown,
    first_not_none,
    placeholder_figure,
    try_twice,
)

# Xiplot has a slightly different layout than slisemap_interactive (mainly different theme)
FIG_LAYOUT = dict(margin=dict(l=10, r=10, t=30, b=10, autoexpand=True), uirevision=True)


class LabelledControls(FlexRow):
    def __init__(
        self,
        kwargs: Dict[str, Any] = {},
        **controls: Any,
    ):
        """Wrap controls in a `FlexRow` with labels on top

        Args:
            **controls: `{label: control}`.
            kwargs: Additional key word arguments forwarded to `FlexRow`
        """
        children = [
            html.Div([lab, ctrl], style={"flex": "1", "minWidth": "12rem"})
            for lab, ctrl in controls.items()
        ]
        super().__init__(*children, **kwargs)


def plugin_load() -> Dict[str, Callable[[Any], pd.DataFrame]]:
    """Xiplot plugin for reading Slisemap files.

    Returns:
        parser: Function for parsing a Slisemap file to a dataframe.
        extension: File extension.
    """
    # TODO Some columns should probably be hidden from the normal plots

    def load(data, max_n: int = 5000, max_l: int = 200):
        return slisemap_to_dataframe(data, max_n=max_n, index=False, losses=max_l)

    return load, ".sm"


class SlisemapEmbeddingPlot(APlot):
    @classmethod
    def name(cls) -> str:
        return "Slisemap embedding plot"

    @classmethod
    def help(cls) -> str:
        return (
            "Plot the embedding of a Slisemap object\n\n"
            + 'Hover over a point when the color is based on "Local loss" to see alternative embeddings for that point.'
        )

    @classmethod
    def register_callbacks(cls, app, df_from_store, df_to_store):
        PdfButton.register_callback(app, cls.get_id(None))

        @app.callback(
            Output(cls.get_id(MATCH), "figure"),
            Input(STORE_DATAFRAME_ID, "data"),
            Input(cls.get_id(MATCH, "variable"), "value"),
            Input(cls.get_id(MATCH, "cluster"), "value"),
            Input(cls.get_id(MATCH, "jitter"), "value"),
            Input(STORE_HOVERED_ID, "data"),
            State(STORE_CLICKED_ID, "data"),
        )
        def callback(df, variable, cluster, jitter, hover, click):
            df = df_from_store(df)
            if cluster in df.columns:
                variable = cluster
            if variable not in df.columns:
                return placeholder_figure(f"{variable} not found")
            dimensions = filter(lambda c: c[:2] == "Z_", df.columns)
            try:
                x = next(dimensions)
                y = next(dimensions)
            except StopIteration:
                return placeholder_figure("Slisemap embedding not found")
            if hover is None:
                hover = click
            return EmbeddingPlot.plot(
                df, x, y, variable, jitter, hover, fig_layout=FIG_LAYOUT
            )

        @app.callback(
            Output(STORE_HOVERED_ID, "data"),
            Input(cls.get_id(ALL), "hoverData"),
            prevent_initial_call=True,
        )
        def hover_callback(inputs):
            return first_not_none(inputs, EmbeddingPlot.get_hover_index)

        @app.callback(
            Output(STORE_CLICKED_ID, "data"),
            Input(cls.get_id(ALL), "clickData"),
            State(STORE_CLICKED_ID, "data"),
            prevent_initial_call=True,
        )
        def click_callback(inputs, old):
            new = first_not_none(inputs, EmbeddingPlot.get_hover_index)
            if new != old:
                return new
            return None

        PlotData.register_callback(
            cls.name(),
            app,
            dict(
                variable=Input(cls.get_id(MATCH, "variable"), "value"),
                cluster=Input(cls.get_id(MATCH, "cluster"), "value"),
                jitter=Input(cls.get_id(MATCH, "jitter"), "value"),
            ),
        )

    @classmethod
    def create_layout(cls, index, df, columns, config=dict()) -> List[Any]:
        return [
            dcc.Graph(id=cls.get_id(index), clear_on_unhover=True),
            LabelledControls(
                Variable=VariableDropdown(
                    df,
                    id=cls.get_id(index, "variable"),
                    value=config.get("variable", None),
                ),
                Clusters=ClusterDropdown(
                    df,
                    id=cls.get_id(index, "cluster"),
                    value=config.get("cluster", None),
                ),
                Jitter=JitterSlider(
                    id=cls.get_id(index, "jitter"),
                    value=config.get("jitter", 0.0),
                    className="stretch",
                ),
            ),
        ]


class SlisemapModelBarPlot(APlot):
    @classmethod
    def name(cls) -> str:
        return "Slisemap barplot for local models"

    @classmethod
    def help(cls) -> str:
        return (
            "Local models from a Slisemap object in a bar plot\n\n"
            + "The coefficients from the local models are plotted in a bar plot. "
            + "Hover over a point in an embedding to see the local model for that point. "
            + "Or use clustering to show the mean models for the clusters."
        )

    @classmethod
    def register_callbacks(cls, app, df_from_store, df_to_store):
        PdfButton.register_callback(app, cls.get_id(None))

        @app.callback(
            Output(cls.get_id(MATCH), "figure"),
            Input(STORE_DATAFRAME_ID, "data"),
            Input(cls.get_id(MATCH, "cluster"), "value"),
            Input(cls.get_id(MATCH, "grouping"), "value"),
            Input(STORE_HOVERED_ID, "data"),
            State(STORE_CLICKED_ID, "data"),
        )
        def callback(df, clusters, grouping, hover, click):
            df = df_from_store(df)
            bs = [c for c in df.columns if c[:2] == "B_"]
            if len(bs) == 0:
                return placeholder_figure("Slisemap local models not found")
            if hover is None:
                hover = click
            return try_twice(
                lambda: ModelBarPlot.plot(
                    df, bs, clusters, grouping, hover, fig_layout=FIG_LAYOUT
                )
            )

        @app.callback(
            Output(cls.get_id(MATCH, "grouping"), "disabled"),
            Input(cls.get_id(MATCH, "cluster"), "value"),
            prevent_initial_call=False,
        )
        def callback(cluster):
            return cluster is None

        PlotData.register_callback(
            cls.name(),
            app,
            dict(
                cluster=Input(cls.get_id(MATCH, "cluster"), "value"),
                grouping=Input(cls.get_id(MATCH, "grouping"), "value"),
            ),
        )

    @classmethod
    def create_layout(cls, index, df, columns, config=dict()) -> List[Any]:
        return [
            dcc.Graph(cls.get_id(index)),
            LabelledControls(
                Clusters=ClusterDropdown(
                    df,
                    id=cls.get_id(index, "cluster"),
                    value=config.get("cluster", None),
                ),
                Grouping=BarGroupingDropdown(
                    id=cls.get_id(index, "grouping"),
                    value=config.get("grouping", None),
                ),
            ),
        ]


class SlisemapModelMatrixPlot(APlot):
    @classmethod
    def name(cls) -> str:
        return "Slisemap matrixplot for local models"

    @classmethod
    def help(cls) -> str:
        return (
            "Local models from a Slisemap object in a matrix plot\n\n"
            + "Hover over a column to see information about that point in other plots."
        )

    @classmethod
    def register_callbacks(cls, app, df_from_store, df_to_store):
        PdfButton.register_callback(app, cls.get_id(None))

        @app.callback(
            Output(cls.get_id(MATCH), "figure"),
            Input(STORE_DATAFRAME_ID, "data"),
            Input(STORE_HOVERED_ID, "data"),
            State(STORE_CLICKED_ID, "data"),
        )
        def callback(df, hover, click):
            df = df_from_store(df)
            try:
                zs0 = next(filter(lambda c: c[:2] == "Z_", df.columns))
            except StopIteration:
                return placeholder_figure("Slisemap embedding not found")
            bs = [c for c in df.columns if c[:2] == "B_"]
            if len(bs) == 0:
                return placeholder_figure("Slisemap local models not found")
            if hover is None:
                hover = click
            return ModelMatrixPlot.plot(df, bs, zs0, hover, fig_layout=FIG_LAYOUT)

        @app.callback(
            Output(STORE_HOVERED_ID, "data"),
            Input(cls.get_id(ALL), "hoverData"),
            prevent_initial_call=True,
        )
        def hover_callback(inputs):
            return first_not_none(inputs, ModelMatrixPlot.get_hover_index)

        @app.callback(
            Output(STORE_CLICKED_ID, "data"),
            Input(cls.get_id(ALL), "clickData"),
            State(STORE_CLICKED_ID, "data"),
            prevent_initial_call=True,
        )
        def click_callback(inputs, old):
            new = first_not_none(inputs, ModelMatrixPlot.get_hover_index)
            if new != old:
                return new
            return None

    @classmethod
    def create_layout(cls, index, df, columns, config=dict()) -> List[Any]:
        return [dcc.Graph(id=cls.get_id(index), clear_on_unhover=True)]


class SlisemapDensityPlot(APlot):
    @classmethod
    def name(cls) -> str:
        return "Slisemap density plot"

    # @classmethod
    # def help(cls) -> str:
    #     return (
    #         "Density plot for Slisemap objects\n\n"
    #         + "Use clustering to easily compare the distribution of the values between different clusters."
    #     )

    @classmethod
    def register_callbacks(cls, app, df_from_store, df_to_store):
        PdfButton.register_callback(app, cls.get_id(None))

        @app.callback(
            Output(cls.get_id(MATCH), "figure"),
            Input(STORE_DATAFRAME_ID, "data"),
            Input(cls.get_id(MATCH, "variable"), "value"),
            Input(cls.get_id(MATCH, "cluster"), "value"),
            Input(STORE_HOVERED_ID, "data"),
            State(STORE_CLICKED_ID, "data"),
        )
        def callback(df, variable, cluster, hover, click):
            df = df_from_store(df)
            if variable not in df.columns:
                return placeholder_figure(f"{variable} not found")
            if hover is None:
                hover = click
            return DistributionPlot.plot(
                df, variable, "Density", cluster, hover, fig_layout=FIG_LAYOUT
            )

        PlotData.register_callback(
            cls.name(),
            app,
            dict(
                variable=Input(cls.get_id(MATCH, "variable"), "value"),
                cluster=Input(cls.get_id(MATCH, "cluster"), "value"),
            ),
        )

    @classmethod
    def create_layout(cls, index, df, columns, config=dict()):
        return [
            dcc.Graph(cls.get_id(index)),
            LabelledControls(
                Variable=VariableDropdown(
                    df,
                    id=cls.get_id(index, "variable"),
                    value=config.get("variable", None),
                ),
                Clusters=ClusterDropdown(
                    df,
                    id=cls.get_id(index, "cluster"),
                    value=config.get("cluster", None),
                ),
            ),
        ]


class SlisemapHistogramPlot(APlot):
    @classmethod
    def name(cls) -> str:
        return "Slisemap histogram plot"

    # @classmethod
    # def help(cls) -> str:
    #     return "Histogram for Slisemap objects"

    @classmethod
    def register_callbacks(cls, app, df_from_store, df_to_store):
        PdfButton.register_callback(app, cls.get_id(None))

        @app.callback(
            Output(cls.get_id(MATCH), "figure"),
            Input(STORE_DATAFRAME_ID, "data"),
            Input(cls.get_id(MATCH, "variable"), "value"),
            Input(cls.get_id(MATCH, "cluster"), "value"),
            Input(STORE_HOVERED_ID, "data"),
            State(STORE_CLICKED_ID, "data"),
        )
        def callback(df, variable, cluster, hover, click):
            df = df_from_store(df)
            if variable not in df.columns:
                return placeholder_figure(f"{variable} not found")
            if hover is None:
                hover = click
            return try_twice(
                lambda: DistributionPlot.plot(
                    df, variable, "Histogram", cluster, hover, fig_layout=FIG_LAYOUT
                )
            )

        PlotData.register_callback(
            cls.name(),
            app,
            dict(
                variable=Input(cls.get_id(MATCH, "variable"), "value"),
                cluster=Input(cls.get_id(MATCH, "cluster"), "value"),
            ),
        )

    @classmethod
    def create_layout(cls, index, df, columns, config=dict()):
        return [
            dcc.Graph(cls.get_id(index)),
            LabelledControls(
                Variable=VariableDropdown(
                    df,
                    id=cls.get_id(index, "variable"),
                    value=config.get("variable", None),
                ),
                Clusters=ClusterDropdown(
                    df,
                    id=cls.get_id(index, "cluster"),
                    value=config.get("cluster", None),
                ),
            ),
        ]


class SlisemapLinearImpactPlot(APlot):
    @classmethod
    def name(cls) -> str:
        return "Slisemap linear impact plot"

    @classmethod
    def help(cls) -> str:
        return (
            "Linear impact plot for Slisemap objects\n\n"
            + 'Plot the "impact" of the variables on the prediction.'
            + " The impact is the variable value times the coefficient."
            + " If the local model is a linear model, then the sum of the impact equals the prediction."
            + "\nThis plot assumes that the local model is a linear model."
        )

    @classmethod
    def register_callbacks(cls, app, df_from_store, df_to_store):
        PdfButton.register_callback(app, cls.get_id(None))

        @app.callback(
            Output(cls.get_id(MATCH), "figure"),
            Input(STORE_DATAFRAME_ID, "data"),
            Input(cls.get_id(MATCH, "pred"), "value"),
            Input(STORE_HOVERED_ID, "data"),
            State(STORE_CLICKED_ID, "data"),
        )
        def callback(df, pred, hover, click):
            df = df_from_store(df)
            if pred is None:
                return placeholder_figure("Could not find prediction")
            if pred not in df.columns:
                return placeholder_figure(f"Could not find prediction '{pred}'")
            if hover is None:
                hover = click
            return try_twice(LinearImpact.plot, df, pred, hover, fig_layout=FIG_LAYOUT)

        PlotData.register_callback(
            cls.name(), app, dict(pred=Input(cls.get_id(MATCH, "pred"), "value"))
        )

    @classmethod
    def create_layout(cls, index, df, columns, config=dict()):
        return [
            dcc.Graph(
                cls.get_id(index), figure=placeholder_figure("Select an item to show")
            ),
            LabelledControls(
                Target=PredictionDropdown(
                    df,
                    id=cls.get_id(index, "pred"),
                    value=config.get("pred", None),
                )
            ),
        ]
