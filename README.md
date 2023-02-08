# [Slisemap](https://github.com/edahelsinki/slisemap)-interactive

Interactive plots for [Slisemap](https://github.com/edahelsinki/slisemap) using [Dash](https://dash.plotly.com/). This package can be used in four different ways:

1. __Standalone:__ To start a standalone dash app just run `slisemap_interactive path/to/slisemap/object.sm` (if the package has been installed) or `python -m slisemap_interactive path/to/slisemap/object.sm` (in the root of this repository).

2. __REPL:__ Use it as replacement (interactive) plots for Slisemap from a Python terminal. Just import the *plot* function, `from slisemap.interactive import plot`, and use it, `plot(slisemap_object)`.

3. __jupyter:__ To create plots in a jupyter notebook, import the *plot* function, `from slisemap.interactive import plot`, and use it, `plot(slisemap_object)`.


4. __[χiplot](https://github.com/edahelsinki/xiplot):__ As a plugin that adds loading and plotting of Slisemap objects.
To use the plugin, just install the package in the same Python environment as [χiplot](https://github.com/edahelsinki/xiplot). *(This has not yet been implemented!)*

## Development

When developing `slisemap_interactive` it is a good idea to do an “[editable](https://setuptools.pypa.io/en/latest/userguide/development_mode.html)” install.
An editable install just links “installed” python files to the python files in this directory (so that any changes are immediately available).
This takes care of the namespace resolution and also activates the "[entry_points](https://setuptools.pypa.io/en/latest/userguide/entry_point.html)" for the *standalone CLI* and *χiplot-plugin*.
To do an editable installation run:

``` pip install --editable . ```

