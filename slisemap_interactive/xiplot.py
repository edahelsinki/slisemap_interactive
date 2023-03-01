"""
    Hooks for connecting to xiplot (using entry points in 'pyproject.toml').
"""
from abc import abstractclassmethod
from typing import Any, Callable, Dict, List, Optional

from xiplot.plugin import (
    APlot,
    placeholder_figure,
    STORE_HOVERED_ID,
    STORE_DATAFRAME_ID,
    PdfButton,
)
from pandas import DataFrame
from dash import html, dcc, Output, Input, MATCH, ALL

from slisemap_interactive.load import slisemap_to_dataframe
from slisemap_interactive.plots import (
    ClusterDropdown,
    DistributionPlot,
    EmbeddingPlot,
    JitterSlider,
    ModelBarDropdown,
    ModelBarPlot,
    ModelMatrixPlot,
    VariableDropdown,
    first_not_none,
)

# TODO Some columns should probably be hidden from the normal plots
# TODO We cannot export data+plots when a Slisemap object is loaded


# Xiplot has a slightly different layout than slisemap_interactive (mainly different theme)
DEFAULT_FIG_LAYOUT = dict(
    margin=dict(l=10, r=10, t=30, b=10, autoexpand=True), uirevision=True
)


def plugin_load() -> Dict[str, Callable[[Any], DataFrame]]:
    """Xiplot plugin for reading Slisemap files.

    Returns:
        parser: Function for parsing a Slisemap file to a dataframe.
        extension: File extension.
    """
    return slisemap_to_dataframe, ".sm"


class SlisemapEmbeddingPlot(APlot):
    @classmethod
    def name(cls) -> str:
        return "Slisemap embedding plot"

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
            prevent_initial_call=False,
        )
        def callback(df, variable, cluster, jitter, hover):
            df = df_from_store(df)
            dimensions = filter(lambda c: c[:2] == "Z_", df.columns)
            try:
                x = next(dimensions)
                y = next(dimensions)
            except StopIteration:
                return placeholder_figure("Slisemap embedding not found")
            if cluster in df.columns:
                variable = cluster
            if variable not in df.columns:
                return placeholder_figure(f"{variable} not found")
            return EmbeddingPlot.plot(
                df, x, y, variable, jitter, hover, fig_layout=DEFAULT_FIG_LAYOUT
            )

        @app.callback(
            Output(STORE_HOVERED_ID, "data"),
            Input(cls.get_id(ALL), "hoverData"),
            prevent_initial_call=True,
        )
        def hover_callback(inputs):
            return first_not_none(inputs, EmbeddingPlot.get_hover_index)

    @classmethod
    def create_layout(cls, index, df, columns, config=dict()) -> List[Any]:
        return [
            dcc.Graph(id=cls.get_id(index), clear_on_unhover=True),
            VariableDropdown(0, df=df, id=cls.get_id(index, "variable")),
            ClusterDropdown(0, df=df, id=cls.get_id(index, "cluster")),
            JitterSlider(0, id=cls.get_id(index, "jitter")),
        ]


class SlisemapModelBarPlot(APlot):
    @classmethod
    def name(cls) -> str:
        return "Slisemap barplot for local models"

    @classmethod
    def register_callbacks(cls, app, df_from_store, df_to_store):
        PdfButton.register_callback(app, cls.get_id(None))

        @app.callback(
            Output(cls.get_id(MATCH), "figure"),
            Input(STORE_DATAFRAME_ID, "data"),
            Input(cls.get_id(MATCH, "cluster"), "value"),
            Input(cls.get_id(MATCH, "grouping"), "value"),
            Input(STORE_HOVERED_ID, "data"),
            prevent_initial_call=False,
        )
        def callback(df, clusters, grouping, hover):
            df = df_from_store(df)
            bs = [c for c in df.columns if c[:2] == "B_"]
            if len(bs) == 0:
                return placeholder_figure("Slisemap local models not found")
            return ModelBarPlot.plot(
                df, bs, clusters, grouping, hover, fig_layout=DEFAULT_FIG_LAYOUT
            )

    @classmethod
    def create_layout(cls, index, df, columns, config=dict()) -> List[Any]:
        return [
            dcc.Graph(cls.get_id(index)),
            ClusterDropdown(0, df=df, id=cls.get_id(index, "cluster")),
            ModelBarDropdown(0, id=cls.get_id(index, "grouping")),
        ]


class SlisemapModelMatrixPlot(APlot):
    @classmethod
    def name(cls) -> str:
        return "Slisemap matrixplot for local models"

    @classmethod
    def register_callbacks(cls, app, df_from_store, df_to_store):
        PdfButton.register_callback(app, cls.get_id(None))

        @app.callback(
            Output(cls.get_id(MATCH), "figure"),
            Input(STORE_DATAFRAME_ID, "data"),
            Input(STORE_HOVERED_ID, "data"),
            prevent_initial_call=False,
        )
        def callback(df, hover):
            df = df_from_store(df)
            try:
                zs0 = next(filter(lambda c: c[:2] == "Z_", df.columns))
            except StopIteration:
                return placeholder_figure("Slisemap embedding not found")
            bs = [c for c in df.columns if c[:2] == "B_"]
            if len(bs) == 0:
                return placeholder_figure("Slisemap local models not found")
            return ModelMatrixPlot.plot(
                df, bs, zs0, hover, fig_layout=DEFAULT_FIG_LAYOUT
            )

        @app.callback(
            Output(STORE_HOVERED_ID, "data"),
            Input(cls.get_id(ALL), "hoverData"),
            prevent_initial_call=True,
        )
        def hover_callback(inputs):
            return first_not_none(inputs, ModelMatrixPlot.get_hover_index)

    @classmethod
    def create_layout(cls, index, df, columns, config=dict()) -> List[Any]:
        return [dcc.Graph(id=cls.get_id(index), clear_on_unhover=True)]


class SlisemapDensityPlot(APlot):
    @classmethod
    def name(cls) -> str:
        return "Slisemap density plot"

    @classmethod
    def register_callbacks(cls, app, df_from_store, df_to_store):
        PdfButton.register_callback(app, cls.get_id(None))

        @app.callback(
            Output(cls.get_id(MATCH), "figure"),
            Input(STORE_DATAFRAME_ID, "data"),
            Input(cls.get_id(MATCH, "variable"), "value"),
            Input(cls.get_id(MATCH, "cluster"), "value"),
            Input(STORE_HOVERED_ID, "data"),
            prevent_initial_call=False,
        )
        def callback(df, variable, cluster, hover):
            df = df_from_store(df)
            if variable not in df.columns:
                return placeholder_figure(f"{variable} not found")
            return DistributionPlot.plot(
                df, variable, "Density", cluster, hover, fig_layout=DEFAULT_FIG_LAYOUT
            )

    @classmethod
    def create_layout(cls, index, df, columns, config=dict()) -> List[Any]:
        return [
            dcc.Graph(cls.get_id(index)),
            VariableDropdown(0, df=df, id=cls.get_id(index, "variable")),
            ClusterDropdown(0, df=df, id=cls.get_id(index, "cluster")),
        ]


class SlisemapHistogramPlot(APlot):
    @classmethod
    def name(cls) -> str:
        return "Slisemap histogram plot"

    @classmethod
    def register_callbacks(cls, app, df_from_store, df_to_store):
        PdfButton.register_callback(app, cls.get_id(None))

        @app.callback(
            Output(cls.get_id(MATCH), "figure"),
            Input(STORE_DATAFRAME_ID, "data"),
            Input(cls.get_id(MATCH, "variable"), "value"),
            Input(cls.get_id(MATCH, "cluster"), "value"),
            Input(STORE_HOVERED_ID, "data"),
            prevent_initial_call=False,
        )
        def callback(df, variable, cluster, hover):
            df = df_from_store(df)
            if variable not in df.columns:
                return placeholder_figure(f"{variable} not found")
            return DistributionPlot.plot(
                df, variable, "Histogram", cluster, hover, fig_layout=DEFAULT_FIG_LAYOUT
            )

    @classmethod
    def create_layout(cls, index, df, columns, config=dict()) -> List[Any]:
        return [
            dcc.Graph(cls.get_id(index)),
            VariableDropdown(0, df=df, id=cls.get_id(index, "variable")),
            ClusterDropdown(0, df=df, id=cls.get_id(index, "cluster")),
        ]
