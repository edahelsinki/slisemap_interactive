"""Slisemap - Interactive: A Dash app for interactively visualising Slisemap objects.
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
"""  # noqa: D205

from slisemap_interactive.app import (  # noqa: F401
    BackgroundApp,
    ForegroundApp,
    plot,
    shutdown,
)
from slisemap_interactive.load import load, slisemap_to_dataframe  # noqa: F401
