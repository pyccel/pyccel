"""
Hook specifications for the pyccel plugin system.

All functions in this module are decorated with ``@hookspec`` and define
the interface that pyccel plugins may implement.  Plugin authors should
decorate their implementations with :data:`pyccel.plugins.hookimpl`.
"""

import argparse
from types import FunctionType
from typing import Iterable, Optional

import pluggy

hookspec = pluggy.HookspecMarker("pyccel")


@hookspec
def get_description() -> str:
    """
    Return a one-line description of the LineAnnot plugin for the CLI help text.

    Return a one-line description of the LineAnnot plugin for the CLI help text.

    Returns
    -------
    str
        Human-readable description of what the plugin does.
    """


@hookspec
def add_cli_options(parser: argparse.ArgumentParser, cli_tool: str):
    """
    Add extra options to the command line tools.

    Add extra options, specific to the plugin, to the command line tools.
    Options should be added to the argument parser, preferably in a new group.

    Parameters
    ----------
    parser : ArgumentParser
        The argument parser to which any options should be added.
    cli_tool : str
        The name of the tool being used.
    """


@hookspec
def read_cli_arguments(kwargs: dict):
    """
    Read any arguments from the kwargs dictionary.

    Read any arguments added via `add_cli_options` from the kwargs dictionary.
    Any arguments relevant to the plugin should be removed from the kwargs so
    the rest of the code cannot be influenced by the plugin. Beware kwargs can
    also come from epyccel so default values may be missing.

    Parameters
    ----------
    kwargs : dict
        The keyword arguments passed to the pipeline.
    """


@hookspec
def get_extra_accelerators() -> Iterable[str]:
    """
    Return the names of any additional accelerators required by the plugin.

    Return the names of any additional accelerators required by the plugin.
    These accelerators affect the compilation of the code and should appear
    in the compiler configurations.

    This hook is optional. It should only be implemented if the plugin needs it.

    Returns
    -------
    Iterable[str]
        Names of the accelerators made available by the plugin
        (e.g. ``['openmp']``).
    """


@hookspec
def get_updated_syntactic_methods() -> Iterable[FunctionType]:
    """
    Return methods to be added to or to override methods in the syntactic parser class.

    The returned callables are injected into a dynamically created subclass
    of the base syntactic parser, so each callable must accept `self` as
    its first argument. The name of the method will remain the same once it
    is added to the class.

    This hook is optional. It should only be implemented if the plugin needs it.

    Returns
    -------
    Iterable[FunctionType]
        Functions to be added or to override syntactic methods.
    """


@hookspec
def get_updated_semantic_methods() -> Iterable[FunctionType]:
    """
    Return methods to be added to or to override methods in the semantic parser class.

    The returned callables are injected into a dynamically created subclass
    of the base semantic parser, so each callable must accept `self` as
    its first argument. The name of the method will remain the same once it
    is added to the class.

    This hook is optional. It should only be implemented if the plugin needs it.

    Returns
    -------
    Iterable[FunctionType]
        Functions to be added or to override semantic methods.
    """


@hookspec
def get_codegen_class(language: str):
    """
    Return the code-generation class to use for language.

    Return a code-generation class which handles translation to the specified
    language. This method can be used to add support for a new language.

    This hook is optional. It should only be implemented if the plugin needs it.

    Parameters
    ----------
    language : str
        The target language (e.g. 'cuda').

    Returns
    -------
    type
        The code-generation class for the requested language or None if this
        plugin doesn't implement the requested language.
    """


@hookspec
def get_updated_codegen_methods(language: str) -> Iterable[FunctionType]:
    """
    Return methods to be added to or to override methods in the code-generation class.

    Return methods to be added to or to override methods in the code-generation class
    for language. The returned callables are injected into a dynamically created
    subclass of the base code-generation class, so each callable must accept `self` as
    its first argument. The name of the method will remain the same once it
    is added to the class.

    This hook is optional. It should only be implemented if the plugin needs it.

    Parameters
    ----------
    language : str
        The target language (e.g. 'c', 'fortran').

    Returns
    -------
    Iterable[FunctionType]
        Functions to be added or to override codegen methods.
    """


@hookspec
def get_wrapper_codegen_class(language: str):
    """
    Return the code-generation class to use for printing the wrapper in language.

    Return a code-generation class which handles translation to the specified
    language. This method can be used to add support for a new language.

    This hook is optional. It should only be implemented if the plugin needs it.

    Parameters
    ----------
    language : str
        The target language (e.g. 'cuda').

    Returns
    -------
    type
        The code-generation class for the requested language or None if this
        plugin doesn't implement the requested language.
    """


@hookspec
def get_updated_wrapper_codegen_methods(language: str) -> Iterable[FunctionType]:
    """
    Return methods to be added to or to override methods in the code-generation class for wrapping.

    Return methods to be added to or to override methods in the code-generation class
    for language. The returned callables are injected into a dynamically created
    subclass of the base code-generation class, so each callable must accept `self` as
    its first argument. The name of the method will remain the same once it
    is added to the class.

    This hook is optional. It should only be implemented if the plugin needs it.

    Parameters
    ----------
    language : str
        The target language (e.g. 'c', 'fortran').

    Returns
    -------
    Iterable[FunctionType]
        Functions to be added or to override codegen methods.
    """


@hookspec
def get_wrapper_class(start_language: str) -> Optional[tuple[type, str]]:
    """
    Return the wrapper class for converting code from start_language to target_language.

    Return a wrapper class which converts code from start_language to
    target_language. This method can be used to add support for a new language.

    This hook is optional. It should only be implemented if the plugin needs it.

    Parameters
    ----------
    start_language : str
        The source language of the wrapper (e.g. 'fortran').

    Returns
    -------
    wrapper_class : type
        The wrapper class for the requested language set or None if this
        plugin doesn't implement the requested language translation.
    target_language : str
        The target language of the wrapper (e.g. 'python').
    """


@hookspec
def get_updated_wrapper_methods(
    start_language: str, target_language: str
) -> Iterable[FunctionType]:
    """
    Return methods to be added or overridden in the wrapper class for a language pair.

    Return methods to be added to or to override methods in a wrapper class
    which converts code from start_language to target_language.
    The returned callables are injected into a dynamically created
    subclass of the base wrapper class, so each callable must accept `self` as
    its first argument. The name of the method will remain the same once it
    is added to the class.

    This hook is optional. It should only be implemented if the plugin needs it.

    Parameters
    ----------
    start_language : str
        The source language of the wrapper (e.g. 'fortran').
    target_language : str
        The target language of the wrapper (e.g. 'python').

    Returns
    -------
    Iterable[FunctionType]
        Functions to be added or to override wrapper methods.
    """


@hookspec
def get_build_generation_class(build_gen_method: str):
    """
    Return the build-generation class for the requested build method.

    Return a build-generation class which handles building with the specified
    framework. This method can be used to add support for a new build framework.

    This hook is optional. It should only be implemented if the plugin needs it.

    Parameters
    ----------
    build_gen_method : str
        The name of the build generation method (e.g. 'cmake').

    Returns
    -------
    type
        The build-generation class for the requested method or None if this
        plugin doesn't implement the requested method.
    """


@hookspec
def get_updated_build_generation_methods(
    build_gen_method: str,
) -> Iterable[FunctionType]:
    """
    Return methods to be added or overridden in the build-generation class.

    Return methods to be added to or to override methods in the build-generation
    class. The returned callables are injected into a dynamically created
    subclass of the base build generation class, so each callable must accept `self` as
    its first argument. The name of the method will remain the same once it
    is added to the class.

    This hook is optional. It should only be implemented if the plugin needs it.

    Parameters
    ----------
    build_gen_method : str
        The name of the build generation method (e.g. 'cmake').

    Returns
    -------
    Iterable[FunctionType]
        Functions to be added or to override build-generation methods.
    """
