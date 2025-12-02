#!/usr/bin/env python
# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#

import sys
import os
import argparse
import pathlib

from .argparse_helpers import add_basic_functionalities, add_compiler_selection, add_accelerator_selection
from .argparse_helpers import check_file_type, add_common_settings

__all__ = ['pyccel']

def setup_pyccel_parser(parser):
    # ... Positional arguments
    group = parser.add_argument_group('Positional arguments')
    group.add_argument('filename', metavar='FILE', type=check_file_type(('.py','.json')),
                        help='Path (relative or absolute) to the Python file to be translated.')

    #... Help and Version
    add_basic_functionalities(parser)
    # ...

    # ... backend compiler options
    group = parser.add_argument_group('Backend selection')

    group.add_argument('--language', choices=('Fortran', 'C', 'Python'), default='Fortran',
                       help='Target language for translation, i.e. the main language of the generated code (default: Fortran).',
                       type=str.lower)

    # ... Compiler options
    add_compiler_selection(parser)

    # ... compiler syntax, semantic and codegen
    group = parser.add_argument_group('Pyccel compiling stages')
    group.add_argument('-x', '--syntax-only', action='store_true',
                       help='Stop Pyccel after syntactic parsing, before semantic analysis or code generation.')
    group.add_argument('-e', '--semantic-only', action='store_true',
                       help='Stop Pyccel after semantic analysis, before code generation.')
    group.add_argument('-t', '--convert-only', action='store_true',
                       help='Stop Pyccel after translation to the target language, before build.')
    # ...

    # ... Additional compiler options
    group = parser.add_argument_group('Additional compiler options')
    group.add_argument('--flags', type=str, \
                       help='Additional compiler flags.')
    group.add_argument('--wrapper-flags', type=str, \
                       help='Additional compiler flags for the wrapper.')
    group.add_argument('--debug', action=argparse.BooleanOptionalAction, default=None,
                        help='Compile the code with debug flags, or not.\n' \
                        ' Overrides the environment variable PYCCEL_DEBUG_MODE, if it exists. Otherwise default is False.')
    group.add_argument('--include',
                        type=str,
                        nargs='*',
                        dest='include',
                        default=(),
                        help='Additional include directories.')
    group.add_argument('--libdir',
                        type=str,
                        nargs='*',
                        dest='libdir',
                        default=(),
                        help='Additional library directories.')
    group.add_argument('--libs',
                        type=str,
                        nargs='*',
                        dest='libs',
                        default=(),
                        help='Additional libraries to link with.')
    group.add_argument('--output', type=pathlib.Path, default = None,\
                       help="Folder in which the output is stored (default: FILE's folder).")
    # ...

    # ... Accelerators
    add_accelerator_selection(parser)
    # ...

    # ... Other options
    group = parser.add_argument_group('Other options')
    add_common_settings(group)
    group.add_argument('--export-compiler-config', action='store_true', deprecated=True,
                        help='Export all compiler information to a JSON file with the given path (relative or absolute).')
    # ...

#==============================================================================
def pyccel_compile_command() -> None:
    """
    Pyccel console command.

    The Pyccel console command allows translating Python files using Pyccel in
    a command-line environment. This function takes no parameters and sets up
    an argument parser for the Pyccel command line interface.

    The command line interface requires a Python file to be specified, and it
    supports various options such as specifying the output language (C,
    Fortran, or Python), compiler settings, and flags for accelerators like
    MPI, OpenMP, and OpenACC. It also includes options for verbosity,
    debugging, and exporting compile information. Unless the user requires the
    process to stop after a specific stage, Pyccel will execute the full
    translation and compilation process until a C Python extension module is
    generated, which can then be imported in Python. In addition, if the input
    file contains an `if __name__ == '__main__':` block, an executable will be
    generated for the corresponding block of code.
    """

    parser = argparse.ArgumentParser(description="Pyccel's command line interface.",
                      add_help=False)

    setup_pyccel_parser(parser)

    # ...
    args = parser.parse_args()
    # ...

    pyccel_compile_command(**vars(args))


def pyccel(*, filename, language, output, export_compiler_config, **kwargs):
    # Imports
    from pyccel.errors.errors     import Errors, PyccelError
    from pyccel.codegen.pipeline  import execute_pyccel

    errors = Errors()
    # ...
    cext = filename.suffix
    if export_compiler_config:
        if cext == '':
            filename = filename.with_suffix('.json')
        if cext != '.json':
            # severity is error to avoid needing to catch exception
            errors.report('Wrong file extension. Expecting `json`, but found',
                          symbol=cext,
                          severity='error')
        else:
            execute_pyccel('',
                           compiler_family = str(compiler) if compiler is not None else None,
                           compiler_export_file = filename)
            sys.exit(0)
    elif cext != '.py':
        # severity is error to avoid needing to catch exception
        errors.report('Wrong file extension. Expecting `py`, but found',
                      symbol=cext,
                      severity='error')
    # ...

    if language == 'python' and output == '':
        errors.report("Cannot output Python file to same folder as this would overwrite the original file. Please specify --output",
                      severity='error')

    errors.check()

    if errors.has_errors():
        sys.exit(1)

    try:
        execute_pyccel(str(filename),
                       language        = language,
                       folder          = output or filename.parent,
                       **kwargs)
    except PyccelError:
        errors.check()
        sys.exit(1)

    errors.check()
    sys.exit(0)
