# Slisemap plugin for χiplot

Plugin for [χiplot](https://github.com/edahelsinki/xiplot) that adds loading and plotting of [SLISEMAP](https://github.com/edahelsinki/slisemap) objects.
To use the plugin, just install it in the same Python environment as χiplot (it should be automatically picked up).

This plugin can also be used standalone. To start a standalone dash app just run `slisemap-dash path/to/slisemap/object.sm` (if the package has been installed) or `python xiplot_slisemap/app.py path/to/slisemap/object.sm` (in the root of the repository).

## TODO

- [x] Slisemap object to dataframe
- [x] Standalone Dash app
- [ ] χiplot plugin
