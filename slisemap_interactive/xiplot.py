"""
    Hooks for connecting to xiplot (using entry points in 'pyproject.toml').
"""
from abc import abstractclassmethod
from typing import Any, Callable, Dict, List, Optional

from xiplot.plots import Plot
from xiplot.utils.layouts import delete_button
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

# TODO some color legends have white text on a bright background
# TODO the plots have a transparent background
# TODO there should probably be a way for plugins to specify dcc.Store:s and general callbacks
# TODO Some columns should probably be hidden from the normal plots
# TODO what happens if we use Slisemap plots on non-Slisemap data
# TODO We cannot export data+plots when a Slisemap object is loaded
# TODO The height of the boxes are too large


def delete_plot_button(index):
    # Why do I have to specify "plot-delete"?
    return delete_button("plot-delete", index)


def pdf_download_button(index):
    # Why is this not handled like the `delete_button` (with a global callback)?
    # TODO in the callback, "figure" should probably be `State` instead of `Input`
    return html.Button(
        "Download as pdf", id={"type": "download_pdf_btn", "index": index}
    )


class APlot(Plot):
    # Suggestion for an improved abstract plot class

    @classmethod
    def get_id(cls, index: Any, subtype: Optional[str] = None) -> Dict[str, Any]:
        # Generate id:s with less typos by using a method
        if subtype is None:
            classtype = f"{cls.__module__}_{cls.__qualname__}"
        else:
            classtype = f"{cls.__module__}_{cls.__qualname__}_{subtype}"
        classtype = classtype.replace(".", "_")
        return {"type": classtype, "index": index}

    @abstractclassmethod
    def register_callbacks(cls, app, df_from_store, df_to_store):
        # Note the change from abstractstaticmethod to abstractclassmethod (to be able to use the new `cls.get_id(index)`)
        pass

    @classmethod
    def create_new_layout(cls, index, df, columns, config=dict()) -> html.Div:
        # There are some standard elements in all plots, these could be handled centrally.
        # Override this to get full control over the layout
        layout = cls.create_layout(index, df, columns, config)
        return html.Div(
            [delete_plot_button(index), pdf_download_button(index)] + layout,
            id=cls.get_id(index, "container"),
            className="plots",
        )

    @abstractclassmethod
    def create_layout(cls, index, df, columns, config=dict()) -> List[Any]:
        # Override this to create a "standard" layout, by only specifying the children
        pass


class SlisemapEmbeddingPlot(APlot):
    @classmethod
    def name(cls) -> str:
        return "Slisemap embedding plot"

    @classmethod
    def register_callbacks(cls, app, df_from_store, df_to_store):
        @app.callback(
            Output(cls.get_id(MATCH), "figure"),
            Input("data_frame_store", "data"),
            Input(cls.get_id(MATCH, "variable"), "value"),
            Input(cls.get_id(MATCH, "cluster"), "value"),
            Input(cls.get_id(MATCH, "jitter"), "value"),
            Input("lastly_hovered_point_store", "data"),
            prevent_initial_call=False,
        )
        def callback(df, variable, cluster, jitter, hover):
            df = df_from_store(df)
            dimensions = filter(lambda c: c[:2] == "Z_", df.columns)
            x = next(dimensions)
            y = next(dimensions)
            if cluster in df.columns:
                variable = cluster
            return EmbeddingPlot.plot(df, x, y, variable, jitter, hover)

        @app.callback(
            Output("lastly_hovered_point_store", "data"),
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
        @app.callback(
            Output(cls.get_id(MATCH), "figure"),
            Input("data_frame_store", "data"),
            Input(cls.get_id(MATCH, "cluster"), "value"),
            Input(cls.get_id(MATCH, "grouping"), "value"),
            Input("lastly_hovered_point_store", "data"),
            prevent_initial_call=False,
        )
        def callback(df, clusters, grouping, hover):
            df = df_from_store(df)
            return ModelBarPlot.plot(df, clusters, grouping, hover)

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
        @app.callback(
            Output(cls.get_id(MATCH), "figure"),
            Input("data_frame_store", "data"),
            Input("lastly_hovered_point_store", "data"),
            prevent_initial_call=False,
        )
        def callback(df, hover):
            df = df_from_store(df)
            return ModelMatrixPlot.plot(df, hover)

        @app.callback(
            Output("lastly_hovered_point_store", "data"),
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
        @app.callback(
            Output(cls.get_id(MATCH), "figure"),
            Input("data_frame_store", "data"),
            Input(cls.get_id(MATCH, "variable"), "value"),
            Input(cls.get_id(MATCH, "cluster"), "value"),
            Input("lastly_hovered_point_store", "data"),
            prevent_initial_call=False,
        )
        def callback(df, variable, cluster, hover):
            df = df_from_store(df)
            return DistributionPlot.plot(df, variable, "Density", cluster, hover)

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
        @app.callback(
            Output(cls.get_id(MATCH), "figure"),
            Input("data_frame_store", "data"),
            Input(cls.get_id(MATCH, "variable"), "value"),
            Input(cls.get_id(MATCH, "cluster"), "value"),
            Input("lastly_hovered_point_store", "data"),
            prevent_initial_call=False,
        )
        def callback(df, variable, cluster, hover):
            df = df_from_store(df)
            return DistributionPlot.plot(df, variable, "Histogram", cluster, hover)

    @classmethod
    def create_layout(cls, index, df, columns, config=dict()) -> List[Any]:
        return [
            dcc.Graph(cls.get_id(index)),
            VariableDropdown(0, df=df, id=cls.get_id(index, "variable")),
            ClusterDropdown(0, df=df, id=cls.get_id(index, "cluster")),
        ]


def plugin_load() -> Dict[str, Callable[[Any], DataFrame]]:
    """Xiplot plugin for reading Slisemap files.

    Returns:
        parser: Function for parsing a Slisemap file to a dataframe.
        extension: File extension.
    """
    return slisemap_to_dataframe, ".sm"


def plugin_embeddingplot():
    return SlisemapEmbeddingPlot


def plugin_barplot():
    return SlisemapModelBarPlot


def plugin_matrixplot():
    return SlisemapModelMatrixPlot


def plugin_densityplot():
    return SlisemapDensityPlot


def plugin_histogramplot():
    return SlisemapHistogramPlot
