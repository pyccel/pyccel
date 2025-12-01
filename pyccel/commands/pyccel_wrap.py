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

from .argparse_helpers import add_basic_functionalities, add_accelerator_selection

__all__ = ['pyccel_wrap_command']

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
                      add_help=False)

    # ... Positional arguments
    group = parser.add_argument_group('Positional arguments')
    group.add_argument('filename', metavar='FILE', type=pathlib.Path,
                       help='Path (relative or absolute) to the Python stub file describing the low-level code.')
    #...

    #... Help and Version
    add_basic_functionalities(parser)
    # ...

    # ... backend compiler options
    group = parser.add_argument_group('Backend selection')

    group.add_argument('--language', choices=('Fortran', 'C'), default='Fortran',
                       help='The language of the code being exposed to Python.',
                       type=str.title)

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
    group.add_argument('-v', '--verbose', action='count', default = 0,\
                        help='Increase output verbosity (use -v, -vv, -vvv for more detailed output).')
    group.add_argument('--developer-mode', action='store_true', \
                        help='Show internal messages.')
    group.add_argument('--time-execution', action='store_true', \
                        help='Print the time spent in each section of the execution.')
    group.add_argument('--conda-warnings', choices=('off', 'basic', 'verbose'),
                        help='Specify the level of Conda warnings to display (default: basic).')
    # ...

    # ...
    args = parser.parse_args()
    # ...

    # Imports
    from pyccel.errors.errors     import Errors, PyccelError
    from pyccel.errors.errors     import ErrorsMode
    from pyccel.errors.messages   import INVALID_FILE_DIRECTORY, INVALID_FILE_EXTENSION
    from pyccel.codegen.wrap_pipeline  import execute_pyccel_wrap

    # ...
    filename = args.filename
    compiler = args.compiler_config or args.compiler_family
    mpi      = args.mpi
    openmp   = args.openmp
    openacc  = False  # [YG 17.06.2025] OpenACC is not supported yet
    output   = args.output or filename.parent

    if not args.conda_warnings:
        args.conda_warnings = 'basic'

    # ...
    # ... report error
    if filename.is_file():
        fext = filename.suffix
        if fext != '.pyi':
            errors = Errors()
            # severity is error to avoid needing to catch exception
            errors.report(INVALID_FILE_EXTENSION,
                            symbol=fext,
                            severity='error')
            errors.check()
            sys.exit(1)
    else:
        # we use Pyccel error manager, although we can do it in other ways
        errors = Errors()
        # severity is error to avoid needing to catch exception
        errors.report(INVALID_FILE_DIRECTORY,
                        symbol=filename,
                        severity='error')
        errors.check()
        sys.exit(1)
    # ...

    accelerators = args.accelerators

    # ...

    # ...
    # this will initialize the singleton ErrorsMode
    # making this setting available everywhere
    err_mode = ErrorsMode()
    if args.developer_mode:
        err_mode.set_mode('developer')
    else:
        err_mode.set_mode(os.environ.get('PYCCEL_ERROR_MODE', 'user'))
    # ...

    try:
        # TODO: prune options
        execute_pyccel_wrap(filename,
                       convert_only    = args.convert_only,
                       verbose         = args.verbose,
                       time_execution  = args.time_execution,
                       language        = args.language.lower(),
                       compiler_family = str(compiler) if compiler is not None else None,
                       debug           = args.debug,
                       accelerators    = accelerators,
                       folder          = output if output is not None else None,
                       output_name     = filename.stem,
                       conda_warnings  = args.conda_warnings)
    except PyccelError:
        sys.exit(1)
