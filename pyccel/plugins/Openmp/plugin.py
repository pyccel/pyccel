"""
Hook implementations for the OpenMP pyccel plugin.
"""

from .. import hookimpl
from .openmp_4_5 import syntactic

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
    if cli_tool in ('compile', 'make', 'wrap'):
        group = parser.add_argument_group("OpenMP compiler options")
        group.add_argument("--omp_version",
                    choices= [4.5, 5.0],
                type= float,
                default= 4.5,
                help= 'OpenMP version to use')


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
    # TODO
    kwargs.pop('omp_version')


@hookimpl
def get_updated_syntactic_methods():
    return [getattr(syntactic, n) for n in syntactic.__all__]

