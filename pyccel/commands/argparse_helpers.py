#!/usr/bin/env python
# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
import argparse
import pathlib
import sys

__all__ = (
        'file_type',
        'AcceleratorAction',
        )

def file_type(suffixes):
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
            raise argparse.ArgumentTypeError("Wrong file extension. Expecting one of: {', '.join(suffixes)}")
        return path.absolute()
    return check_path

def add_basic_functionalities(parser):
    #... Help and Version
    import pyccel
    version = pyccel.__version__
    libpath = pyccel.__path__[0]
    python  = f'python {sys.version_info.major}.{sys.version_info.minor}'
    message = f'pyccel {version} from {libpath} ({python})'

    group = parser.add_argument_group('Basic options')
    group.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
    group.add_argument('-V', '--version', action='version', help='Show version and exit.', version=message)
    # ...

def add_compiler_selection(parser):
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

