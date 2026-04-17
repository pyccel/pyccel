# ------------------------------------------------------------------------- #
# This file is part of Pyccel which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/pyccel/blob/devel/LICENSE #
# for full license details.                                                 #
# ------------------------------------------------------------------------- #
"""Module containing scripts to handle the pyccel compile sub-command."""

import argparse
import pathlib
import sys

from .argparse_helpers import (
    add_accelerator_selection,
    add_common_settings,
    add_compiler_selection,
    path_with_suffix,
)

__all__ = ("pyccel_compile", "setup_pyccel_compile_parser", "PYCCEL_COMPILE_DESCR")

PYCCEL_COMPILE_DESCR = "Translate and compile a single Python file."


def setup_pyccel_compile_parser(parser):
    """
    Add the `pyccel compile` arguments to the parser.

    Add the `pyccel compile` arguments to the parser for command line arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to be modified.
    """
    # ... Positional arguments
    group = parser.add_argument_group("Positional arguments")

    group.add_argument(
        "filename",
        metavar="FILE",
        type=path_with_suffix((".py",)),
        help="Path (relative or absolute) to the Python file to be translated.",
    )
    # ...

    # ... backend compiler options
    group = parser.add_argument_group("Backend selection")

    group.add_argument(
        "--language",
        choices=("Fortran", "C", "Python"),
        default="Fortran",
        help="Target language for translation, i.e. the main language of the generated code (default: Fortran).",
        type=str.title,
    )

    # ... Compiler options
    add_compiler_selection(parser, allow_compiler_config=True)

    # ... compiler syntax, semantic and codegen
    group = parser.add_argument_group("Pyccel compiling stages")
    group.add_argument(
        "-x",
        "--syntax-only",
        action="store_true",
        help="Stop Pyccel after syntactic parsing, before semantic analysis or code generation.",
    )
    group.add_argument(
        "-e",
        "--semantic-only",
        action="store_true",
        help="Stop Pyccel after semantic analysis, before code generation.",
    )
    group.add_argument(
        "-t",
        "--convert-only",
        action="store_true",
        help="Stop Pyccel after translation to the target language, before build.",
    )
    # ...

    # ... Additional compiler options
    group = parser.add_argument_group("Additional compiler options")
    group.add_argument("--flags", type=str, help="Additional compiler flags.")
    group.add_argument(
        "--wrapper-flags", type=str, help="Additional compiler flags for the wrapper."
    )
    group.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compile the code with debug flags, or not.\n"
        " Overrides the environment variable PYCCEL_DEBUG_MODE, if it exists. Otherwise default is False.",
    )
    group.add_argument(
        "--include",
        type=str,
        nargs="*",
        dest="include",
        default=(),
        help="Additional include directories.",
    )
    group.add_argument(
        "--libdir",
        type=str,
        nargs="*",
        dest="libdir",
        default=(),
        help="Additional library directories.",
    )
    group.add_argument(
        "--libs",
        type=str,
        nargs="*",
        dest="libs",
        default=(),
        help="Additional libraries to link with.",
    )
    group.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Folder in which the output is stored (default: FILE's folder).",
    )
    # ...

    # ... Accelerators
    add_accelerator_selection(parser)
    # ...

    # ... Other options
    group = parser.add_argument_group("Other options")
    add_common_settings(group)
    # ...


def pyccel_compile(*, filename, language, output, **kwargs):
    """
    Call the pyccel pipeline.

    Handle the deprecated --export-compiler-config command and call the pyccel pipeline.

    Parameters
    ----------
    filename : Path
        Name of the Python file to be translated.
    language : str
        The target language Pyccel is translating to.
    output : str
        Path to the working directory.
    **kwargs : dict
        See execute_pyccel.
    """
    # Imports
    from pyccel.codegen.pipeline import execute_pyccel
    from pyccel.errors.errors import Errors

    errors = Errors()
    # ...
    if not filename.is_file():
        errors.report(f"File not found: {filename}", severity="error")

    if language == "Python" and output == "":
        errors.report(
            "Cannot output Python file to same folder as this would overwrite the original file. Please specify --output",
            severity="error",
        )

    if errors.has_errors():
        print(errors, end="")
        sys.exit(1)

    execute_pyccel(
        str(filename),
        language=language.lower(),
        folder=output or filename.parent,
        **kwargs,
    )
