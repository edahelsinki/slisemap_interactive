"""
    Hooks for connecting to xiplot (using entry points in 'pyproject.toml').
"""
from typing import Any, Callable, Dict, List

import pandas as pd
from dash import ALL, MATCH, Input, Output, State, dcc, html
from xiplot.plugin import (
    ID_CLICKED,
    ID_DATAFRAME,
    ID_HOVERED,
    ID_PLOTLY_TEMPLATE,
    APlot,
    FlexRow,
    PdfButton,
    PlotData,
)

from slisemap_interactive.load import (
    DEFAULT_MAX_L,
    DEFAULT_MAX_N,
    slisemap_to_dataframe,
)
from slisemap_interactive.plots import (
    BarGroupingDropdown,
    ClusterDropdown,
    ContourCheckbox,
    DistributionPlot,
    EmbeddingPlot,
    JitterSlider,
    LinearTerms,
    ModelBarPlot,
    ModelMatrixPlot,
    PredictionDropdown,
    VariableDropdown,
    first_not_none,
    placeholder_figure,
    try_twice,
)


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
            html.Div(
                [lab, ctrl],
                style=None
                if isinstance(ctrl, dcc.Checklist)
                else {"flex": "1", "minWidth": "12rem"},
            )
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

    def load(data, max_n: int = DEFAULT_MAX_N, max_l: int = DEFAULT_MAX_L):
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
            Input(ID_DATAFRAME, "data"),
            Input(cls.get_id(MATCH, "variable"), "value"),
            Input(cls.get_id(MATCH, "cluster"), "value"),
            Input(cls.get_id(MATCH, "contours"), "value"),
            Input(cls.get_id(MATCH, "jitter"), "value"),
            Input(ID_HOVERED, "data"),
            State(ID_CLICKED, "data"),
            Input(ID_PLOTLY_TEMPLATE, "data"),
        )
        def callback(df, variable, cluster, contours, jitter, hover, click, template):
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
                df, x, y, variable, contours, jitter, hover, template=template
            )

        @app.callback(
            Output(ID_HOVERED, "data"),
            Input(cls.get_id(ALL), "hoverData"),
            prevent_initial_call=True,
        )
        def hover_callback(inputs):
            return first_not_none(inputs, EmbeddingPlot.get_hover_index)

        @app.callback(
            Output(ID_CLICKED, "data"),
            Input(cls.get_id(ALL), "clickData"),
            State(ID_CLICKED, "data"),
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
                Density=ContourCheckbox(
                    id=cls.get_id(index, "contours"),
                    value=config.get("contours", False),
                    labelClassName="button",
                    labelStyle={"display": "inline-block"},
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
            Input(ID_DATAFRAME, "data"),
            Input(cls.get_id(MATCH, "cluster"), "value"),
            Input(cls.get_id(MATCH, "grouping"), "value"),
            Input(ID_HOVERED, "data"),
            State(ID_CLICKED, "data"),
            Input(ID_PLOTLY_TEMPLATE, "data"),
        )
        def callback(df, clusters, grouping, hover, click, template):
            df = df_from_store(df)
            bs = [c for c in df.columns if c[:2] == "B_"]
            if len(bs) == 0:
                return placeholder_figure("Slisemap local models not found")
            if hover is None:
                hover = click
            return try_twice(
                ModelBarPlot.plot, df, bs, clusters, grouping, hover, template=template
            )

        @app.callback(
            Output(cls.get_id(MATCH, "grouping"), "disabled"),
            Input(cls.get_id(MATCH, "cluster"), "value"),
            prevent_initial_call=False,
        )
        def callback_disabled(cluster):
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
            Input(ID_DATAFRAME, "data"),
            Input(ID_HOVERED, "data"),
            State(ID_CLICKED, "data"),
            Input(ID_PLOTLY_TEMPLATE, "data"),
        )
        def callback(df, hover, click, template):
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
            return ModelMatrixPlot.plot(df, bs, zs0, hover, template=template)

        @app.callback(
            Output(ID_HOVERED, "data"),
            Input(cls.get_id(ALL), "hoverData"),
            prevent_initial_call=True,
        )
        def hover_callback(inputs):
            return first_not_none(inputs, ModelMatrixPlot.get_hover_index)

        @app.callback(
            Output(ID_CLICKED, "data"),
            Input(cls.get_id(ALL), "clickData"),
            State(ID_CLICKED, "data"),
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
            Input(ID_DATAFRAME, "data"),
            Input(cls.get_id(MATCH, "variable"), "value"),
            Input(cls.get_id(MATCH, "cluster"), "value"),
            Input(ID_HOVERED, "data"),
            State(ID_CLICKED, "data"),
            Input(ID_PLOTLY_TEMPLATE, "data"),
        )
        def callback(df, variable, cluster, hover, click, template):
            df = df_from_store(df)
            if variable not in df.columns:
                return placeholder_figure(f"{variable} not found")
            if hover is None:
                hover = click
            return DistributionPlot.plot(
                df, variable, "Density", cluster, hover, template=template
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
            Input(ID_DATAFRAME, "data"),
            Input(cls.get_id(MATCH, "variable"), "value"),
            Input(cls.get_id(MATCH, "cluster"), "value"),
            Input(ID_HOVERED, "data"),
            State(ID_CLICKED, "data"),
            Input(ID_PLOTLY_TEMPLATE, "data"),
        )
        def callback(df, variable, cluster, hover, click, template):
            df = df_from_store(df)
            if variable not in df.columns:
                return placeholder_figure(f"{variable} not found")
            if hover is None:
                hover = click
            return try_twice(
                DistributionPlot.plot,
                df,
                variable,
                "Histogram",
                cluster,
                hover,
                template=template,
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


class SlisemapLinearTermsPlot(APlot):
    @classmethod
    def name(cls) -> str:
        return "Slisemap linear terms plot"

    @classmethod
    def help(cls) -> str:
        return (
            "Linear terms plot for Slisemap objects\n\n"
            + 'Plot the "terms" of the linear models (variables times coefficients).'
            + " If the local model is a linear model, then the sum of the terms equals the prediction."
            + "\nThis plot assumes that the local model is a linear model and that the data has not been unscaled."
        )

    @classmethod
    def register_callbacks(cls, app, df_from_store, df_to_store):
        PdfButton.register_callback(app, cls.get_id(None))

        @app.callback(
            Output(cls.get_id(MATCH), "figure"),
            Input(ID_DATAFRAME, "data"),
            Input(cls.get_id(MATCH, "pred"), "value"),
            Input(ID_HOVERED, "data"),
            State(ID_CLICKED, "data"),
            Input(ID_PLOTLY_TEMPLATE, "data"),
        )
        def callback(df, pred, hover, click, template):
            df = df_from_store(df)
            if pred is None:
                return placeholder_figure("Could not find prediction")
            if pred not in df.columns:
                return placeholder_figure(f"Could not find prediction '{pred}'")
            if hover is None:
                hover = click
            return try_twice(LinearTerms.plot, df, pred, hover, template=template)

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
                Prediction=PredictionDropdown(
                    df,
                    id=cls.get_id(index, "pred"),
                    value=config.get("pred", None),
                )
            ),
        ]
