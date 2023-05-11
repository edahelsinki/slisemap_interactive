"""
    Slisemap - Interactive: A Dash app for interactively visualising Slisemap objects
    =================================================================================

    Use the `plot` function for non-blocking interactive plots in notebooks and interactive interpreters.
    In (non-Python) terminals you can call `slisemap_interactive` to start a standalone application.
    Finally, this package also integrates into χiplot as a plugin.

    Relevant links:
    ---------------

    - [GitHub repository](https://github.com/edahelsinki/slisemap_interactive)
    - [Slisemap](https://github.com/edahelsinki/slisemap)
    - [χiplot](https://github.com/edahelsinki/xiplot)
    - [Dash](https://dash.plotly.com/)
"""

from slisemap_interactive.app import plot, shutdown, BackgroundApp, ForegroundApp
from slisemap_interactive.load import slisemap_to_dataframe


def __version__():
    from importlib.metadata import version

    return version("slisemap_interactive")
