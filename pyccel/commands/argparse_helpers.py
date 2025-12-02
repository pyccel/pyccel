#!/usr/bin/env python
# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
File containing functions that help build argparse.ArgumentParser objects.

File containing functions that help build argparse.ArgumentParser objects. Several of these
functions are common to multiple sub-commands so the logic can be shared.
"""
import argparse
import os
import pathlib
import sys

from pyccel import __version__ as pyccel_version, __path__ as pyccel_path
from pyccel.errors.errors     import ErrorsMode
from pyccel.compilers.default_compilers import available_compilers

__all__ = (
        'add_accelerator_selection',
        'add_common_settings',
        'add_compiler_selection',
        'add_version_flag',
        'check_file_type',
        'ErrorModeSelector',
        )

# -----------------------------------------------------------------------------------------
def check_file_type(suffixes):
    """
    Check if the input is a type with one of the suffixes.

    Check if the input is a type with one of the specified suffixes.

    Parameters
    ----------
    suffixes : iterable[str]
        An iterable describing the valid suffixes.

    Returns
    -------
    function
        A function which checks if the argument is of the expected type.
    """
    def check_path(path_str):
        """
        Check if path_str describes the path to an existing file with the expected suffix.
        """
        path = pathlib.Path(path_str)
        if not path.is_file():
            raise argparse.ArgumentTypeError(f"File not found: {path}")
        if path.suffix not in suffixes:
            raise argparse.ArgumentTypeError(f"Wrong file extension for file: {path}. Expecting one of: {', '.join(suffixes)}")
        return path.absolute()
    return check_path

# -----------------------------------------------------------------------------------------
def add_version_flag(parser):
    """
    Add version flag to argument parser.

    Add version flag to argument parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to be modified.
    """
    version = pyccel_version
    libpath = pyccel_path[0]
    python  = f'python {sys.version_info.major}.{sys.version_info.minor}'
    message = f'pyccel {version} from {libpath} ({python})'

    parser.add_argument('-V', '--version', action='version', help='Show version and exit.', version=message)

# -----------------------------------------------------------------------------------------
def add_compiler_selection(parser):
    """
    Add compiler selection flags to argument parser.

    Add flags to argument parser to select a compiler. This can be done by family
    or with a json config file.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to be modified.
    """
    default_compiler = os.environ.get('PYCCEL_DEFAULT_COMPILER', 'GNU')
    group = parser.add_argument_group('Compiler configuration (mutually exclusive options)')
    compiler_group = group.add_mutually_exclusive_group(required=False)
    compiler_group.add_argument('--compiler-family',
                                dest='compiler_family',
                                choices=available_compilers.keys(),
                                type=str,
                                default=default_compiler,
                                help=f'Compiler family (default: {default_compiler}).')
    compiler_group.add_argument('--compiler-config',
                                dest='compiler_family',
                                type=lambda p: str(check_file_type(('.json',))),
                                default=None,
                                metavar='CONFIG.json',
                                help='Load all compiler information from a JSON file with the given path (relative or absolute).')

# -----------------------------------------------------------------------------------------
def add_accelerator_selection(parser):
    """
    Add accelerator flags to argument parser.

    Add flags to argument parser to select any accelerators to be used in compilation.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to be modified.
    """
    group = parser.add_argument_group('Accelerators options')
    group.add_argument('--mpi', dest='accelerators', action='append_const', const='mpi',
                       default=[], help='Use MPI.')
    group.add_argument('--openmp', dest='accelerators', action='append_const', const='openmp',
                       help='Use OpenMP.')
#    group.add_argument('--openacc', dest='accelerators', action='append_const', const='openacc',
#                       help='Use OpenACC.') # [YG 17.06.2025] OpenACC is not supported yet

# -----------------------------------------------------------------------------------------
def add_common_settings(parser):
    """
    Add common settings controlling how Pyccel reports progress.

    Add settings controlling how Pyccel reports progress:
    - verbosity
    - developer-mode
    - conda warnings level
    - time execution

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to be modified.
    """
    # Set default error mode
    err_mode = ErrorsMode()
    err_mode.set_mode(os.environ.get('PYCCEL_ERROR_MODE', 'user'))

    parser.add_argument('-v', '--verbose', action='count', default = 0,
                        help='Increase output verbosity (use -v, -vv, -vvv for more detailed output).')
    parser.add_argument('--developer-mode', action=ErrorModeSelector, nargs=0, const='developer',
                        help='Show internal messages.', dest=argparse.SUPPRESS)
    parser.add_argument('--conda-warnings', choices=('off', 'basic', 'verbose'), default='basic',
                        help='Specify the level of Conda warnings to display (default: basic).')
    parser.add_argument('--time-execution', action='store_true',
                        help='Print the time spent in each section of the execution.')

# -----------------------------------------------------------------------------------------
class ErrorModeSelector(argparse.Action):
    """
    Class describing an action which sets the error mode.

    Class describing an action which sets the error mode to the value saved in the
    const argument.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        err_mode = ErrorsMode()
        err_mode.set_mode(self.const)
