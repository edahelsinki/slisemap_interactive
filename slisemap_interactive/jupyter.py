"""
Run the interactive plot in a jupyter notebook
"""

from typing import Any, Dict, Literal, Union
from os import PathLike
from warnings import warn

import pandas as pd
from slisemap import Slisemap
from jupyter_dash import JupyterDash

from .load import slisemap_to_dataframe
from .app import setup_page


def _can_display_iframe() -> bool:
    try:
        from IPython import get_ipython

        ipython = str(type(get_ipython()))
        if "google.colab" in ipython:
            return True  # Google colab
        elif "zmqshell" in ipython:
            return True  # Jupyter
        if "terminal" in ipython:
            return False  # IPython console
        else:
            return False  # Unknown
    except:
        return False  # not even IPython


class BackgroundApp(JupyterDash):

    # Store app singleton for reuse
    __app = None

    def __init__(
        self, slisemap: Union[pd.DataFrame, Slisemap, str, PathLike], *args, **kwargs
    ):
        """Create the `JupyterDash` app, see `JupyterDash()`."""
        if BackgroundApp.__app is not None:
            warn(
                "A `JupyterApp` already exists. Use `JupyterApp.get_app(...)` instead of `JupyterApp(...)` to reuse it.",
                Warning,
            )
        super().__init__(*args, **kwargs)
        if not isinstance(slisemap, pd.DataFrame):
            slisemap = slisemap_to_dataframe(slisemap)
        setup_page(self, slisemap)
        self._display_url = None
        self._display_port = None
        self._display_call = None

    @classmethod
    def get_app(
        cls,
        slisemap: Union[pd.DataFrame, Slisemap, str, PathLike],
        appargs: Dict[str, Any] = {},
        runargs: Dict[str, Any] = {},
    ) -> "BackgroundApp":
        """Get the currently running `JupyterApp`, or start a new one.

        Args:
            slisemap: TODO remove
            appargs: Keyword arguments to `JupyterApp()`. Defaults to {}.
            runargs: Keyword arguments to `JupyterApp.run_server()`. Defaults to {}.

        Returns:
            The currently running `JupyterApp`.
        """
        if BackgroundApp.__app is None:
            app = BackgroundApp(slisemap, **appargs)
            app.run_server(**runargs)
        return BackgroundApp.__app

    def run_server(self, *args, **kwargs):
        """Start the server, see `JupyterDash.run_server()`."""
        super().run_server(*args, **kwargs)
        BackgroundApp.__app = self

    def shutdown(self):
        """Shutdown the server"""
        for thread in self._server_threads.values():
            thread.kill()
            thread.join()
        self._server_threads.clear()
        if BackgroundApp.__app == self:
            BackgroundApp.__app = None

    def _display_in_colab(self, url, port, mode, width, height):
        # Catch parameters to the display function for reuse later (see `BackgroundApp.display()`)
        self._display_url = url
        self._display_port = port
        self._display_call = super()._display_in_colab

    def _display_in_jupyter(self, url, port, mode, width, height):
        # Catch parameters to the display function for reuse later (see `BackgroundApp.display()`)
        self._display_url = url
        self._display_port = port
        self._display_call = super()._display_in_jupyter

    def display(
        self,
        mode: Literal[None, "inline", "external", "jupyterlab"] = None,
        width: Union[str, int] = "100%",
        height: Union[str, int] = 1000,
    ):
        """Display the plots.

        Args:
            mode: How should the plot be displayed (see `JupyterDash.run_server()`). Defaults to "inline" in a notebook and to "external" otherwise.
            width: Width of the iframe. Defaults to "100%".
            height: Height of the iframe. Defaults to 1000.

        Raises:
            Exception: The server must be started (through `run_server()`) before the plots are displayed.
        """
        if self._display_call is None:
            raise Exception("You need to run `run_server()` before displaying results")
        if mode is None:
            mode = "inline" if _can_display_iframe() else "external"
        self._display_call(self._display_url, self._display_port, mode, width, height)


def plot(slisemap: Union[pd.DataFrame, Slisemap, str, PathLike], *args, **kwargs):
    """Plot a Slisemap object interactively.
    This is designed to be called from a jupyter notebook or an interactive Python shell.
    This function automatically starts a server in the background.

    Args:
        slisemap: The Slisemap object.
        *args: Positional arguments forwarded to `BackgroundApp.display()`.
        **kwargs: Keyword arguments forwarded to `BackgroundApp.display()`.
    """
    app = BackgroundApp.get_app(slisemap)
    # TODO allow changing of data
    app.display(*args, **kwargs)
