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
        'add_help_flag',
        'add_version_flag',
        'deprecation_warning',
        'get_warning_and_line',
        'path_with_suffix',
        'ErrorModeSelector',
        )

# -----------------------------------------------------------------------------------------
def get_warning_and_line():
    """
    Get colored WARNING and LINE strings.

    Get colored WARNING and LINE strings if termcolor is installed, otherwise return
    plain strings.

    Returns
    -------
    tuple[str, str]
        The WARNING and LINE strings.
    """
    try:
        from termcolor import colored
        WARNING = colored('WARNING', 'red', attrs=['bold', 'blink'])
        LINE    = colored('-------', 'red', attrs=['bold', 'blink'])
    except ImportError:
        WARNING = 'WARNING'
        LINE    = '-------'
    return WARNING, LINE

# -----------------------------------------------------------------------------------------
def deprecation_warning(tool):
    """
    Create a deprecation warning message for an old pyccel-TOOL command.

    Create a deprecation warning message for an old pyccel-TOOL command.

    Parameters
    ----------
    tool : str
        The name of the tool for which pyccel-TOOL is a deprecated command.

    Returns
    -------
    str
        The deprecation warning message.
    """
    WARNING, LINE = get_warning_and_line()
    message = f"{WARNING}: The pyccel-{tool} command is deprecated and will be removed in v 2.3. Please use `pyccel {tool}` instead."
    return "\n".join([LINE, message, LINE])

# -----------------------------------------------------------------------------------------
def path_with_suffix(suffixes):
    """
    Get the function which returns a Path to a file with one of the suffixes.

    Get the function which returns a Path to a file with one of the specified
    suffixes. The function returns a argparse.ArgumentTypeError if the input
    does not respect the expected file format.

    Parameters
    ----------
    suffixes : iterable[str]
        An iterable describing the valid suffixes.

    Returns
    -------
    function
        A function which checks if the argument is of the expected type.
    """
    def convert_to_path_with_suffix(path_str):
        """
        Covert string to Path with the chosen suffix.

        Parameters
        ----------
        path_str : str
            The string passed to the executable and parsed by argparse.

        Returns
        -------
        Path
            The Path describing the parameter.

        Raises
        ------
        argparse.ArgumentTypeError
            Raised if the string does not respect the expected format.
        """
        path = pathlib.Path(path_str)
        if not path.is_file():
            raise argparse.ArgumentTypeError(f"File not found: {path}")
        if path.suffix not in suffixes:
            raise argparse.ArgumentTypeError(f"Wrong file extension for file: {path}. Expecting one of: {', '.join(suffixes)}")
        return path.absolute()
    return convert_to_path_with_suffix

# -----------------------------------------------------------------------------------------
def add_help_flag(parser):
    """
    Add -h/--help flag to argument parser.

    Add -h/--help flag to argument parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to be modified.
    """
    message = 'Show this help message and exit.'
    parser.add_argument('-h', '--help', action='help', help=message)

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
    Add group of compiler selection flags to argument parser.

    Add argument group to parser, with flags for selecting a compiler. This can be
    done by family or with a json config file.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to be modified.
    """
    default_compiler = os.environ.get('PYCCEL_DEFAULT_COMPILER', 'GNU')
    json_file_checker = path_with_suffix(('.json',))

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
                                type=lambda p: str(json_file_checker(p)),
                                default=None,
                                metavar='CONFIG.json',
                                help='Load all compiler information from a JSON file with the given path (relative or absolute).')

# -----------------------------------------------------------------------------------------
def add_accelerator_selection(parser):
    """
    Add group of accelerator flags to argument parser.

    Add argument group to parser, with flags for selecting any accelerators to be used
    in compilation.

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
    - help
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

    add_help_flag(parser)
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

    Parameters
    ----------
    **kwargs : dict
        See argparse.Action.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        err_mode = ErrorsMode()
        err_mode.set_mode(self.const)
