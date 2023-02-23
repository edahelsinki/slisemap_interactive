import sys
from slisemap_interactive.app import cli

# This script is only called when there is a local copy of the repository,
# (since `pyproject.toml` contains a different `entry_point` for "scripts").
# Thus, it is assumed this is used in development / debug mode.
sys.argv.append("--debug")
cli()
