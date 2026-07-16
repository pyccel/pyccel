# Plugin System

Pyccel's plugin system is built on [pluggy](https://pluggy.readthedocs.io/).

## How it works

Available hooks are declared in [`pyccel/plugins/hookspecs.py`](../pyccel/plugins/hookspecs.py). Plugins implement the three hooks required for the command line interface, as well as any additional hooks relevant for the functionality. The helpers in [`pyccel/plugins/plugin_tools.py`](../pyccel/plugins/plugin_tools.py) build augmented `parser/codegen/wrapping` classes by layering plugin methods on top of the base classes.

## Writing a plugin

1. Create a module that imports `hookimpl` from `pyccel` and decorates each hook implementation with it.
2. Publish it as a Python package with the entry-point `pyccel` (see `pluggy` docs and [setuptools docs](https://setuptools.pypa.io/en/latest/userguide/entry_point.html) for more information).

## Built-in plugins

| Plugin | Location | Purpose |
|--------|----------|---------|
| `LineAnnot` | [`pyccel/plugins/LineAnnot/`](../pyccel/plugins/LineAnnot/) | Inserts source-line comments into generated code |

## Note

The set of available hooks is still evolving. Always check [`hookspecs.py`](../pyccel/plugins/hookspecs.py) for the current interface.
