#!/usr/bin/env python
# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
import argparse
import os
import pathlib
import sys

from pyccel import __version__ as pyccel_version, __path__ as pyccel_path
from ..compilers.default_compilers import available_compilers

__all__ = (
        'file_type',
        'AcceleratorAction',
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
            raise argparse.ArgumentTypeError(f"Wrong file extension. Expecting one of: {', '.join(suffixes)}")
        return path.absolute()
    return check_path

# -----------------------------------------------------------------------------------------
def add_basic_functionalities(parser):
    """
    Add basic functionalities to argument parser.

    Add basic functionalities such as a help and a version message to argument parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to be modified.
    """
    version = pyccel_version
    libpath = pyccel_path[0]
    python  = f'python {sys.version_info.major}.{sys.version_info.minor}'
    message = f'pyccel {version} from {libpath} ({python})'

    group = parser.add_argument_group('Basic options')
    group.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
    group.add_argument('-V', '--version', action='version', help='Show version and exit.', version=message)

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
                                choices=available_compilers.keys(),
                                default=default_compiler,
                                help=f'Compiler family (default: {default_compiler}).')
    compiler_group.add_argument('--compiler-config',
                                type=check_file_type(('.json',)),
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
    group.add_argument('--mpi', action=AcceleratorAction, nargs=0,
                       help='Use MPI.')
    group.add_argument('--openmp', action=AcceleratorAction, nargs=0,
                       help='Use OpenMP.')
#    group.add_argument('--openacc', action=AcceleratorAction, nargs=0,
#                       help='Use OpenACC.') # [YG 17.06.2025] OpenACC is not supported yet

# -----------------------------------------------------------------------------------------
class AcceleratorAction(argparse.Action):
    """
    A class to describe the action which groups accelerators into one output.

    A class to describe the action which groups accelerators passed as input
    to argparse via flags (e.g. `--mpi`) into a single list (accelerators=[mpi]).
    """
    def __call__(self, parser, namespace, values, option_string):
        """
        The function called by argparse when the argument is passed.

        This method is invoked automatically by argparse when an argument using
        this action is encountered. It ensures that the name of each accelerator
        flag (e.g., `mpi` or `openmp`) is added to a shared `accelerators`
        list within the parsed arguments namespace.

        Parameters
        ----------
        parser : argparse.ArgumentParser
            The argument parser instance that is processing the command line.
        namespace : argparse.Namespace
            The namespace object holding attributes for all parsed arguments.
        values : NoneType
            The value(s) associated with the argument. For flag arguments with
            `nargs=0`, this will be `None`.
        option_string : str, optional
            The specific option string that triggered this action, such as
            `'--mpi'` or `'--openmp'`.
        """
        # Initialise the list if it doesn't exist yet
        accelerators = getattr(namespace, 'accelerators', [])
        accelerators.append(self.dest)
        setattr(namespace, 'accelerators', accelerators)

