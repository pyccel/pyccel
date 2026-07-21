"""
Hook implementations for the OpenMP pyccel plugin.
"""

from .. import hookimpl
from . import openmp_4_5, openmp_5_0


class OpenMPConfig:
    def __init__(self):
        self._version = None

    @property
    def version(self):
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
    openmp_config.version = kwargs.pop("omp_version", 4.5)
    kwargs.setdefault("accelerators", []).append("openmp")


@hookimpl
def get_updated_syntactic_methods():
    return [
        getattr(openmp_config.syntactic, n) for n in openmp_config.syntactic.__all__
    ]


@hookimpl
def get_updated_semantic_methods():
    return [getattr(openmp_config.semantic, n) for n in openmp_config.semantic.__all__]


@hookimpl
def get_updated_codegen_methods(language: str):
    if language == "fortran":
        return [getattr(openmp_config.fcode, n) for n in openmp_config.fcode.__all__]
    elif language == "c":
        return [getattr(openmp_config.ccode, n) for n in openmp_config.ccode.__all__]
    elif language == "python":
        return [getattr(openmp_config.pycode, n) for n in openmp_config.pycode.__all__]
    else:
        return []
