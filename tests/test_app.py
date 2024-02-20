import sys

import pandas as pd
import pytest

from slisemap_interactive.app import BackgroundApp, _can_display_iframe, cli, shutdown
from slisemap_interactive.layout import page_with_all_plots


def test_layout():
    page_with_all_plots(pd.DataFrame(), 0)


def test_background():
    app = BackgroundApp()
    app.set_data(pd.DataFrame())
    with pytest.raises(RuntimeError):
        app.display()
    app.shutdown()
    shutdown()
    _can_display_iframe()


def test_cli_parse(tmp_path):
    old_argv = sys.argv
    sys.argv = [old_argv[0], "--export", str(tmp_path / "out"), str(tmp_path / "in")]
    with pytest.raises(FileNotFoundError):
        cli()
    sys.argv = old_argv
