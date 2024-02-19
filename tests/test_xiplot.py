import inspect
from io import BytesIO
from typing import Any, Callable, Dict, Union, get_args, get_origin

import numpy as np
import pandas as pd
from slisemap import Slisemap
from xiplot.plugin import APlot, AReadPlugin

from slisemap_interactive.plots import JitterSlider
from slisemap_interactive.xiplot import (
    LabelledControls,
    SlisemapDensityPlot,
    SlisemapEmbeddingPlot,
    SlisemapHistogramPlot,
    SlisemapLinearTermsPlot,
    SlisemapModelBarPlot,
    SlisemapModelMatrixPlot,
    plugin_load,
)


def unoptional(s: object) -> object:
    """Unwrap Optional signature."""
    is_optional = (
        get_origin(s) == Union
        and len(get_args(s)) == 2
        and get_args(s)[1] == type(None)
    )
    return get_args(s)[0] if is_optional else s


def check_annotation(ann1: Dict[str, object], ann2: Dict[str, object]) -> bool:
    """Naive check that annotations match."""
    ann1 = unoptional(ann1)
    ann2 = unoptional(ann2)
    if ann1 == ann2:
        return True
    if ann1 in [Any, object] or ann2 in [Any, object]:
        return True
    if get_origin(ann1) == get_origin(ann2):
        return all(
            check_annotation(s1i, s2i)
            for s1i, s2i in zip(get_args(ann1), get_args(ann2))
        )
    return False


def assert_annotation_match(f1: Callable, ann2: Dict[str, object]):
    """Naive assert that the annotation of function `f1` matches the annotation `ann2`."""
    s1 = f1.__annotations__
    s1.setdefault("return", None)
    ann2.setdefault("return", None)
    assert s1.keys() == ann2.keys(), f"{f1.__qualname__}: {s1} != {ann2}"
    for i1, i2 in zip(s1.items(), ann2.items()):
        assert check_annotation(i1, i2)


def signature_to_annotation(f: Callable) -> Dict[str, object]:
    """Convert a signature into an annotation-like dictionary."""
    fsign = inspect.signature(f)
    ann = {k: v.annotation for k, v in fsign.parameters.items()}
    ann.setdefault("return", fsign.return_annotation)
    return ann


def type_to_annotation(typ: type, reference: Callable):
    """Convert a typing hint into an annotation-like dictionary."""
    fsign = inspect.signature(reference)
    assert len(fsign.parameters) == len(get_args(typ)[0])
    ann = dict(zip(fsign.parameters, get_args(typ)[0]))
    ann.setdefault("return", get_args(typ)[1])
    return ann


def test_load_signature():
    assert_annotation_match(plugin_load, type_to_annotation(AReadPlugin, plugin_load))
    assert_annotation_match(plugin_load, signature_to_annotation(plugin_load))


def test_load():
    X = np.random.normal(0, 1, (10, 3))
    Y = np.random.normal(0, 1, 10)
    B0 = np.random.normal(0, 1, (10, 4))
    sm = Slisemap(X, Y, lasso=0.1, B0=B0)
    with BytesIO() as io:
        sm.save(io)
        io.seek(0)
        sm2 = plugin_load()[0](io)
    assert sm2.shape == (10, 30)


def test_plot_signature():
    plots = [
        SlisemapDensityPlot,
        SlisemapEmbeddingPlot,
        SlisemapHistogramPlot,
        SlisemapLinearTermsPlot,
        SlisemapModelBarPlot,
        SlisemapModelMatrixPlot,
    ]
    for plot in plots:
        assert issubclass(plot, APlot)
        for name in plot.__dict__:
            if name[0] != "_":
                method = getattr(plot, name)
                assert_annotation_match(method, getattr(APlot, name).__annotations__)
                assert_annotation_match(method, signature_to_annotation(method))


def test_plots():
    plots = [
        SlisemapDensityPlot,
        SlisemapEmbeddingPlot,
        SlisemapHistogramPlot,
        SlisemapLinearTermsPlot,
        SlisemapModelBarPlot,
        SlisemapModelMatrixPlot,
    ]
    for plot in plots:
        plot.create_layout(0, pd.DataFrame(), None, {})


def test_labelled_controls():
    LabelledControls({"id": "test"}, test=JitterSlider(id="jitter"))
