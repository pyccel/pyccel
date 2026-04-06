import argparse
import pluggy


hookspec = pluggy.HookspecMarker("pyccel")


@hookspec
def get_description() -> str:
    """
    Get a description of the plugin for use in the CLI.
    """


@hookspec
def add_cli_options(parser : argparse.ArgumentParser, cli_tool : str):
    """
    Add options to the command line tools.

    Parameters
    ----------
    parser : ArgumentParser
        The argument parser to which any options should be added
        (preferably in a new group).
    cli_tool : str
        The name of the tool being used.
    """


@hookspec
def get_extra_accelerators():
    """
    """


@hookspec
def get_updated_syntactic_methods():
    """
    """


@hookspec
def get_updated_semantic_methods():
    """
    """


@hookspec
def get_updated_ccode_methods():
    """
    """


@hookspec
def get_updated_fcode_methods():
    """
    """


@hookspec
def get_updated_c_to_python_wrapper_methods():
    """
    """


@hookspec
def get_updated_fortran_to_c_wrapper_methods():
    """
    """


@hookspec
def get_updated_cmake_gen_methods():
    """
    """


@hookspec
def get_updated_meson_gen_methods():
    """
    """
