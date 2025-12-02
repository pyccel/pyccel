#!/usr/bin/env python
# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
File containing the pyccel-wrap command line interface.
"""

import sys
import os
import argparse
import pathlib

from .argparse_helpers import add_version_flag, add_accelerator_selection
from .argparse_helpers import check_file_type, add_common_settings

__all__ = ['pyccel_wrap_command']

PYCCEL_WRAP_DESCR = 'Create the wrapper to allow code to be called from Python.'

def setup_pyccel_wrap_parser(parser):
    """
    Add the pyccel-wrap arguments to the parser.

    Add the pyccel-wrap arguments to the parser for command line arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to be modified.
    """
    # ... Positional arguments
    group = parser.add_argument_group('Positional arguments')
    group.add_argument('filename', metavar='FILE', type=check_file_type(('.pyi',)),
                       help='Path (relative or absolute) to the Python stub file describing the low-level code.')
    #...

    # ... backend compiler options
    group = parser.add_argument_group('Backend selection')

    group.add_argument('--language', choices=('Fortran', 'C'), default='Fortran',
                       help='The language of the code being exposed to Python.',
                       type=str.lower)

    # ... Compiler options
    group = parser.add_argument_group('Compiler configuration (mutually exclusive options)')
    compiler_group = group.add_mutually_exclusive_group(required=False)
    compiler_group.add_argument('--compiler-family',
                                type=str,
                                default=os.environ.get('PYCCEL_DEFAULT_COMPILER', 'GNU'),
                                metavar='FAMILY',
                                help='Compiler family {GNU,intel,PGI,nvidia,LLVM} (default: GNU).')
    compiler_group.add_argument('--compiler-config',
                                type=pathlib.Path,
                                default=None,
                                metavar='CONFIG.json',
                                help='Load all compiler information from a JSON file with the given path (relative or absolute).')

    # ... Additional compiler options
    group = parser.add_argument_group('Additional compiler options')
    group.add_argument('--debug', action=argparse.BooleanOptionalAction, default=None,
                        help='Compile the code with debug flags, or not.\n' \
                        ' Overrides the environment variable PYCCEL_DEBUG_MODE, if it exists. Otherwise default is False.')

    group.add_argument('--output', type=pathlib.Path, default = None,\
                       help="Folder in which the output is stored (default: FILE's folder).")
    # ...

    # ... Accelerators
    add_accelerator_selection(parser)
    # ...

    # ... Other options
    group = parser.add_argument_group('Other options')
    group.add_argument('-t', '--convert-only', action='store_true',
                       help='Stop Pyccel after generating the wrapper files but before building the Python extension file.')
    add_common_settings(group)

#==============================================================================
def pyccel_wrap_command() -> None:
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
                      add_help=True)

    #... Help and Version
    add_version_flag(parser)
    # ...
    # ...
    setup_pyccel_wrap_parser(parser)
    # ...
    args = parser.parse_args()
    # ...

    print("warning: The pyccel-wrap command is deprecated and will be removed in v2.3. Please use pyccel wrap.", file=sys.stderr)

    try:
        pyccel_wrap(**vars(args))
    except PyccelError:
        pass

    print(errors)
    sys.exit(errors.has_errors())

def pyccel_wrap(*, filename, output, **kwargs) -> None:
    # Imports
    from pyccel.errors.errors     import PyccelError
    from pyccel.codegen.wrap_pipeline  import execute_pyccel_wrap
    # ...

    execute_pyccel_wrap(filename,
                   folder          = output or filename.parent,
                   **kwargs)
