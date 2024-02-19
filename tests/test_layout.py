import pandas as pd

from slisemap_interactive.layout import page_with_all_plots


def test_layout():
    page_with_all_plots(pd.DataFrame(), 0)
