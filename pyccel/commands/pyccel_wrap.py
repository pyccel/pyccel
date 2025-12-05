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

from .argparse_helpers import add_version_flag, add_accelerator_selection, add_compiler_selection
from .argparse_helpers import path_with_suffix, add_common_settings

__all__ = ('pyccel_wrap_command',
           'pyccel_wrap')

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
    group.add_argument('filename', metavar='FILE', type=path_with_suffix(('.pyi',)),
                       help='Path (relative or absolute) to the Python stub file describing the low-level code.')
    #...

    # ... backend compiler options
    group = parser.add_argument_group('Backend selection')

    group.add_argument('--language', choices=('Fortran', 'C'), default='Fortran',
                       help='The language of the code being exposed to Python.',
                       type=str.title)

    # ... Compiler options
    add_compiler_selection(parser)

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
    Command line wrapper for the deprecated `pyccel-wrap` command line tool.

    Command line wrapper for the deprecated `pyccel-wrap` command line tool.
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

    print("Warning: The pyccel-wrap command is deprecated and will be removed in v2.3. Please use `pyccel wrap` instead.", file=sys.stderr)

    from pyccel.errors.errors     import Errors, PyccelError
    from pyccel.utilities.stage   import PyccelStage

    pyccel_stage = PyccelStage()
    errors = Errors()

    try:
        pyccel_wrap(**vars(args))
    except PyccelError:
        pass

    pyccel_stage.pyccel_finished()
    print(errors, end='')
    sys.exit(errors.has_errors())

def pyccel_wrap(*, filename, language, output, **kwargs) -> None:
    """
    Call the `pyccel wrap` pipeline.

    Import and call the `pyccel wrap` pipeline.

    Parameters
    ----------
    filename : Path
        Name of the Python file to be translated.
    language : str
        The target language Pyccel is translating to.
    output : str
        Path to the working directory.
    **kwargs : dict
        See execute_pyccel_wrap.
    """
    # Imports
    from pyccel.codegen.wrap_pipeline  import execute_pyccel_wrap
    # ...

    execute_pyccel_wrap(filename,
                        language = language.lower(),
                        folder = output or filename.parent,
                        **kwargs)
