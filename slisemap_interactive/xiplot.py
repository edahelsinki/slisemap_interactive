"""
    Hooks for connecting to xiplot (using entry points in 'pyproject.toml').
"""
from typing import Any, Callable, Dict, List

from xiplot.plugin import (
    APlot,
    placeholder_figure,
    STORE_HOVERED_ID,
    STORE_DATAFRAME_ID,
    PdfButton,
    PlotData,
)
import pandas as pd
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

# Xiplot has a slightly different layout than slisemap_interactive (mainly different theme)
DEFAULT_FIG_LAYOUT = dict(
    margin=dict(l=10, r=10, t=30, b=10, autoexpand=True), uirevision=True
)


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
            VariableDropdown(
                df, id=cls.get_id(index, "variable"), value=config.get("variable", None)
            ),
            ClusterDropdown(
                df, id=cls.get_id(index, "cluster"), value=config.get("cluster", None)
            ),
            JitterSlider(
                id=cls.get_id(index, "jitter"), value=config.get("jitter", 0.0)
            ),
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
            ClusterDropdown(
                df, id=cls.get_id(index, "cluster"), value=config.get("cluster", None)
            ),
            ModelBarDropdown(
                id=cls.get_id(index, "grouping"), value=config.get("grouping", None)
            ),
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
            VariableDropdown(
                df, id=cls.get_id(index, "variable"), value=config.get("variable", None)
            ),
            ClusterDropdown(
                df, id=cls.get_id(index, "cluster"), value=config.get("cluster", None)
            ),
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
            VariableDropdown(
                df, id=cls.get_id(index, "variable"), value=config.get("variable", None)
            ),
            ClusterDropdown(
                df, id=cls.get_id(index, "cluster"), value=config.get("cluster", None)
            ),
        ]
