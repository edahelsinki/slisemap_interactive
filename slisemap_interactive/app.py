"""
    Simple standalone Dash app.
"""
import argparse
import os
import sys
from os import PathLike
from typing import Any, Dict, Literal, Union
from warnings import warn

import pandas as pd
from dash import Dash
from jupyter_dash import JupyterDash
from slisemap import Slisemap

from slisemap_interactive.layout import register_callbacks, page_with_all_plots
from slisemap_interactive.load import slisemap_to_dataframe
from slisemap_interactive.plots import DataCache


def cli():
    """Plot a slisemap object interactively. This function acts like a command line program."""
    parser = argparse.ArgumentParser(
        prog="slisemap-interactive",
        description="Slisemap - Interactive:   A Dash app for interactively visualising Slisemap objects",
    )
    parser.add_argument(
        "PATH",
        help="The path to a Slisemap object (or a directory containing a Slisemap object)",
    )
    parser.add_argument("--host", help="Host IP used to serve the application")
    parser.add_argument("-p", "--port", help="Port used to serve the application")
    parser.add_argument("-d", "--debug", action="store_true", help="enable debug mode")
    args = parser.parse_args()
    path = args.PATH
    if os.path.isdir(path):
        for path in [f for f in os.listdir(path) if f.endswith(".sm")]:
            print("Using:", path)
            break
    kwargs = {}
    if args.debug:
        kwargs["debug"] = True
    if args.host:
        kwargs["host"] = args.host
    if args.port:
        kwargs["port"] = args.port
    ForegroundApp().set_data(path).run(**kwargs)


def plot(slisemap: Union[pd.DataFrame, Slisemap, str, PathLike], *args, **kwargs):
    """Plot a Slisemap object interactively.
    This function is designed to be called from a jupyter notebook or an interactive Python shell.
    This function automatically starts a server in the background.

    Args:
        slisemap: The Slisemap object.
        *args: Positional arguments forwarded to `BackgroundApp.display()`.
        **kwargs: Keyword arguments forwarded to `BackgroundApp.display()`.
    """
    BackgroundApp.get_app().set_data(slisemap).display(*args, **kwargs)


class ForegroundApp(Dash):
    """Create a blocking Dash app for interactive visualisations of a Slisemap object."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, title="Interactive Slisemap", **kwargs)
        self.data_cache = DataCache()
        register_callbacks(self, self.data_cache)

    def set_data(
        self, slisemap: Union[pd.DataFrame, Slisemap, str, PathLike]
    ) -> "ForegroundApp":
        """Set which data the app should show new connections.
        Old data is cached so that old connections continue working.
        For existing connections, refresh the page to get the latest data.

        Args:
            slisemap: Data.

        Returns:
            self for chaining.
        """
        if not isinstance(slisemap, pd.DataFrame):
            slisemap = slisemap_to_dataframe(slisemap, max_n=5000)
        key = self.data_cache.add_data(slisemap)
        self.layout = page_with_all_plots(slisemap, key)
        return self


class BackgroundApp(JupyterDash):
    """Create a non-blocking Dash app for interactive visualisations of Slisemap objects."""

    # Store current app for reuse as a singleton
    __app = None

    def __init__(self, *args, **kwargs):
        """Create the `BackgroundApp` server, see `jupyter_dash.JupyterDash()` for arguments."""
        super().__init__(*args, **kwargs)
        self._display_url = None
        self._display_port = None
        self._display_call = None
        self.data_cache = DataCache()
        register_callbacks(self, self.data_cache)

    def set_data(
        self, slisemap: Union[pd.DataFrame, Slisemap, str, PathLike]
    ) -> "BackgroundApp":
        """Set which data the app should show new connections.
        Old data is cached so that old connections continue working.
        For existing connections, refresh the page to get the latest data.

        Args:
            slisemap: Data.

        Returns:
            self for chaining.
        """
        if not isinstance(slisemap, pd.DataFrame):
            slisemap = slisemap_to_dataframe(slisemap, max_n=5000)
        key = self.data_cache.add_data(slisemap)
        self.layout = page_with_all_plots(slisemap, key)
        return self

    @classmethod
    def get_app(
        cls, appargs: Dict[str, Any] = {}, runargs: Dict[str, Any] = {}
    ) -> "BackgroundApp":
        """Get the currently running `JupyterApp`, or start a new one.

        Args:
            appargs: Keyword arguments to `JupyterApp()`. Defaults to {}.
            runargs: Keyword arguments to `JupyterApp.run_server()`. Defaults to {}.

        Returns:
            The currently running `JupyterApp`.
        """
        if BackgroundApp.__app is None:
            app = BackgroundApp(**appargs)
            app.run_server(**runargs)
        return BackgroundApp.__app

    def run_server(self, *args, **kwargs):
        """Start the server, see `JupyterDash.run_server()`."""
        if BackgroundApp.__app is not None:
            warn(
                "A `BackgroundApp` already exists. Use `BackgroundApp.get_app(...)` to reuse it.",
                Warning,
            )
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


def _can_display_iframe() -> bool:
    """Check if the current Python session is able to display iframes."""
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


if __name__ == "__main__":
    sys.argv.append("--debug")
    cli()
