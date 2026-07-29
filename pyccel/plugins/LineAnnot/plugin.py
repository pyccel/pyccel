"""
Hook implementations for the LineAnnot pyccel plugin.
"""

from .. import hookimpl
from .semantic import _visit_CodeBlock


@hookimpl
def get_description():
    """
    Return a one-line description of the LineAnnot plugin for the CLI help text.

    Return a one-line description of the LineAnnot plugin for the CLI help text.

    Returns
    -------
    str
        Human-readable description of what the plugin does.
    """
    return "Add comments in generated code indicating which line in the Python file corresponds to the generated code."


@hookimpl
def add_cli_options(parser, cli_tool):
    """
    Add LineAnnot-specific CLI options to *parser*.

    The LineAnnot plugin requires no extra options beyond the ``--line_annotation``
    flag added automatically by `plugin_tools.get_plugin_cli_options`, so
    this hook is a no-op.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to which any options should be added.
    cli_tool : str
        The name of the CLI tool being invoked.
    """


@hookimpl
def read_cli_arguments(kwargs: dict):
    """
    Read any arguments from the kwargs dictionary.

    The LineAnnot plugin has no specific arguments to read, so this hook is a
    no-op.

    Parameters
    ----------
    kwargs : dict
        The keyword arguments passed to the pipeline.
    """


@hookimpl
def remove_cli_arguments(kwargs: dict):
    """
    Remove any plugin-specific arguments from the kwargs dictionary.

    The LineAnnot plugin adds no arguments, so this hook is a
    no-op.

    Parameters
    ----------
    kwargs : dict
        The keyword arguments passed to the pipeline.
    """


@hookimpl
def get_updated_semantic_methods():
    """
    Return the semantic parser methods provided or overridden by the LineAnnot plugin.

    Return the semantic parser methods provided or overridden by the LineAnnot plugin.

    Returns
    -------
    tuple[function]
        A tuple containing the method overridden by this plugin.
    """
    return (_visit_CodeBlock,)
