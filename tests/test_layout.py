import sys

import pandas as pd
import pytest

from slisemap_interactive.app import BackgroundApp, cli, shutdown
from slisemap_interactive.layout import page_with_all_plots


def test_layout():
    page_with_all_plots(pd.DataFrame(), 0)


def test_background():
    app = BackgroundApp()
    app.set_data(pd.DataFrame())
    app.shutdown()
    shutdown()


def test_cli_parse():
    old_argv = sys.argv
    sys.argv = [old_argv[0], "--export", "", ""]
    with pytest.raises(FileNotFoundError):
        cli()
    sys.argv = old_argv
