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
def get_updated_codegen_methods(language : str):
    """
    """


@hookspec
def get_codegen_class(language : str):
    """
    """


@hookspec
def get_wrapper_class(start_language : str, target_language : str):
    """
    """


@hookspec
def get_updated_wrapper_methods(start_language : str, target_language : str):
    """
    """


@hookspec
def get_build_generation_class(build_gen_method : str):
    """
    """


@hookspec
def get_updated_build_generation_methods(build_gen_method : str):
    """
    """
