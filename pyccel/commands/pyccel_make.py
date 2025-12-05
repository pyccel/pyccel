#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
File containing the pyccel-make command line interface.
"""

import argparse
import glob
import sys
from pathlib import Path

from .argparse_helpers import add_version_flag, add_compiler_selection, add_accelerator_selection
from .argparse_helpers import add_common_settings, check_file_type

PYCCEL_MAKE_DESCR = 'Translate and compile multiple Python files in a project.'

class GlobAction(argparse.Action):
    """
    Class describing a glob argument.

    Class describing a glob argument. The action saves the files matching
    the glob to the specified destination. The type check is used on the
    located files.

    Parameters
    ----------
    option_strings : obj
        See argparse.Action.
    dest : str
        The name of the attribute where the files are saved.
    nargs : int, optional
        The number of arguments that can be passed to this flag.
    type : Function, optional
        A function which can be used to check the type of the files.
    **kwargs : dict
        See argparse.Action.
    """
    def __init__(self, option_strings, dest, nargs=None, type=None, **kwargs): #pylint: disable=redefined-builtin
        self._type_check = type
        super().__init__(option_strings, dest, nargs=None, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        # Get files
        files = [Path(f) for f in glob.glob(values, recursive=True)]

        # Check types
        try:
            for f in files:
                self._type_check(f)
        except argparse.ArgumentTypeError as err:
            raise argparse.ArgumentError(self, message=err) from err

        # Save result
        setattr(namespace, self.dest, files)

class FileDescriptionAction(argparse.Action):
    """
    Class describing an argument which collects files listed in another file.

    Class describing an argument which collects files listed in another file.
    The action saves the listed files to the specified destination. The type
    check is used on the located files.

    Parameters
    ----------
    option_strings : obj
        See argparse.Action.
    dest : str
        The name of the attribute where the files are saved.
    nargs : int, optional
        The number of arguments that can be passed to this flag.
    type : Function, optional
        A function which can be used to check the type of the files.
    **kwargs : dict
        See argparse.Action.
    """
    def __init__(self, option_strings, dest, nargs=None, type=None, **kwargs): #pylint: disable=redefined-builtin
        self._type_check = type
        super().__init__(option_strings, dest, nargs=None, type=Path, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        # Get files
        with open(values, 'r', encoding='utf-8') as f:
            files = [Path(fname.strip()) for fname in f.readlines()]

        # Check types
        try:
            for f in files:
                self._type_check(f)
        except argparse.ArgumentTypeError as err:
            raise argparse.ArgumentError(self, message=err) from err

        # Save result
        setattr(namespace, self.dest, files)

def setup_pyccel_make_parser(parser):
    """
    Add the pyccel-make arguments to the parser.

    Add the pyccel-make arguments to the parser for command line arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to be modified.
    """
    # ...
    group = parser.add_argument_group('File specification',
            description = "Use one of the below methods to specify which files should be translated."
            ).add_mutually_exclusive_group(required=True)
    group.add_argument('-f', '--files', nargs='+', type=check_file_type(('.py',)), metavar='FILE',
            help="A list of files to be translated as a project.")
    group.add_argument('-g', '--glob', dest='files', action=GlobAction, type=check_file_type(('.py',)),
            help=("A glob that should be used to recognise files to be translated as a project (e.g. '**/*.py'). "
                  "Note: quote the pattern to prevent shell expansion."))
    group.add_argument('-d', '--file-descr', dest='files', action=FileDescriptionAction, type=check_file_type(('.py',)),
            help="A UTF-8 text file containing the paths to the files to be translated as a project. One path (relative or absolute) per line.")

    # ... backend compiler options
    group = parser.add_argument_group('Backend selection')

    group.add_argument('--language', choices=('Fortran', 'C', 'Python'), default='Fortran',
                       help='Target language for translation, i.e. the main language of the generated code (default: Fortran).',
                       type=str.title)
    group.add_argument('--build-system', choices=('meson', 'cmake'), default='meson',
                       help='Chosen build system for translation, i.e. the tool that will be used to compile the generated code (default: meson).',
                       type=str.lower)

    # ... Compiler options
    add_compiler_selection(parser)

    # ... Additional compiler options
    group = parser.add_argument_group('Additional compiler options')
    group.add_argument('--debug', action=argparse.BooleanOptionalAction, default=None,
                        help='Compile the code with debug flags, or not.\n' \
                        ' Overrides the environment variable PYCCEL_DEBUG_MODE, if it exists. Otherwise default is False.')
    group.add_argument('--output', type=Path, default = None, dest='folder',
                       help="Folder in which the output is stored (default: FILE's folder).")

    # ... Accelerators
    add_accelerator_selection(parser)
    # ...

    # ... Other options
    group = parser.add_argument_group('Other options')
    add_common_settings(group)
    group.add_argument('-t', '--convert-only', action='store_false', dest='build_code',
                       help='Stop Pyccel after translation to the target language, before build.')

def pyccel_make_command() -> None:
    """
    Pyccel-make console command.

    The command line interface allowing pyccel-make to be called.
    """
    parser = argparse.ArgumentParser(description="Pyccel's command line interface for multi-file projects.",
            add_help = True)

    #... Help and Version
    add_version_flag(parser)

    # ...
    setup_pyccel_make_parser(parser)
    # ...
    args = parser.parse_args()

    print("warning: The pyccel-make command is deprecated and will be removed in v2.3. Please use pyccel make.", file=sys.stderr)

    from pyccel.errors.errors     import Errors, PyccelError
    from pyccel.utilities.stage   import PyccelStage

    pyccel_stage = PyccelStage()
    errors = Errors()

    try:
        pyccel_make(**vars(args))
    except PyccelError:
        pass

    pyccel_stage.pyccel_finished()
    print(errors, end='')
    sys.exit(errors.has_errors())

def pyccel_make(*, language, **kwargs) -> None:
    """
    Call the pyccel make pipeline.

    Import and call the pyccel make pipeline.

    Parameters
    ----------
    language : str
        The target language Pyccel is translating to.
    **kwargs : dict
        See execute_pyccel_make.
    """

    from pyccel.codegen.make_pipeline  import execute_pyccel_make

    execute_pyccel_make(language = language.lower(), **kwargs)
