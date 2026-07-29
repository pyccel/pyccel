"""
Hook implementations for the OpenMP pyccel plugin.
"""

from .. import hookimpl
from . import openmp_4_5, openmp_5_0


class OpenMPConfig:
    """
    A class to store the chosen configuration for the OpenMP plugin.

    A class to store the chosen configuration for the OpenMP plugin.
    """

    def __init__(self):
        self._version = None

    @property
    def version(self):
        """
        Get the OpenMP version.

        Get the OpenMP version.
        """
        return self._version

    @version.setter
    def version(self, v):
        self._version = v
        if v == 4.5:
            mod = openmp_4_5
        elif v == 5.0:
            mod = openmp_5_0
        else:
            raise RuntimeError("Unsupported version")

        self.syntactic = mod.syntactic
        self.semantic = mod.semantic
        self.fcode = mod.fcode
        self.ccode = mod.ccode
        self.pycode = mod.pycode


openmp_config = OpenMPConfig()


@hookimpl
def get_description():
    """
    Return a one-line description of the OpenMP plugin for the CLI help text.

    Return a one-line description of the OpenMP plugin for the CLI help text.

    Returns
    -------
    str
        Human-readable description of what the plugin does.
    """
    return "A plugin to add support for OpenMP parallelisation via comments."


@hookimpl
def add_cli_options(parser, cli_tool):
    """
    Add OpenMP-specific CLI options to *parser*.

    The OpenMP plugin has 1 argument `omp_version` which is only required
    for the tools which eventually compile code.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to which any options should be added.
    cli_tool : str
        The name of the CLI tool being invoked.
    """
    if cli_tool in ("compile", "make", "wrap"):
        group = parser.add_argument_group("OpenMP compiler options")
        group.add_argument(
            "--omp_version",
            choices=[4.5, 5.0],
            type=float,
            default=4.5,
            help="OpenMP version to use",
        )


@hookimpl
def remove_cli_arguments(kwargs: dict):
    """
    Remove any plugin-specific arguments from the kwargs dictionary.

    Remove any arguments added via `add_cli_options` from the kwargs dictionary.
    Any arguments relevant to the plugin should be removed from the kwargs so
    the rest of the code cannot be influenced by the plugin. Beware kwargs can
    also come from epyccel so default values may be missing.

    Parameters
    ----------
    kwargs : dict
        The keyword arguments passed to the pipeline.
    """
    kwargs.pop("omp_version", 4.5)


@hookimpl
def read_cli_arguments(kwargs: dict):
    """
    Read any arguments from the kwargs dictionary.

    Read the version from the kwargs dictionary and update the
    accelerators.

    Parameters
    ----------
    kwargs : dict
        The keyword arguments passed to the pipeline.
    """
    openmp_config.version = kwargs.pop("omp_version", 4.5)
    kwargs.setdefault("accelerators", []).append("openmp")


@hookimpl
def get_updated_syntactic_methods():
    """
    Return methods to be added to or to override methods in the syntactic parser class.

    Return methods to be added to or to override methods in the syntactic parser class.

    Returns
    -------
    Iterable[FunctionType]
        Functions to be added or to override syntactic methods.
    """
    return [
        getattr(openmp_config.syntactic, n) for n in openmp_config.syntactic.__all__
    ]


@hookimpl
def get_updated_semantic_methods():
    """
    Return methods to be added to or to override methods in the semantic parser class.

    Return methods to be added to or to override methods in the semantic parser class.

    Returns
    -------
    Iterable[FunctionType]
        Functions to be added or to override semantic methods.
    """
    return [getattr(openmp_config.semantic, n) for n in openmp_config.semantic.__all__]


@hookimpl
def get_updated_codegen_methods(language: str):
    """
    Return methods to be added to or to override methods in the code-generation class.

    Return methods to be added to or to override methods in the code-generation class.

    Parameters
    ----------
    language : str
        The target language (e.g. 'c', 'fortran').

    Returns
    -------
    Iterable[FunctionType]
        Functions to be added or to override codegen methods.
    """
    if language == "fortran":
        return [getattr(openmp_config.fcode, n) for n in openmp_config.fcode.__all__]
    elif language == "c":
        return [getattr(openmp_config.ccode, n) for n in openmp_config.ccode.__all__]
    elif language == "python":
        return [getattr(openmp_config.pycode, n) for n in openmp_config.pycode.__all__]
    else:
        return []
