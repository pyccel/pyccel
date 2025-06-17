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

__all__ = ['MyParser', 'pyccel']

#==============================================================================
class MyParser(argparse.ArgumentParser):
    """
    Custom argument parser for printing help message in case of an error.
    See http://stackoverflow.com/questions/4042452/display-help-message-with-python-argparse-when-script-is-called-without-any-argu
    """
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

#==============================================================================
def pyccel() -> None:
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

    parser = MyParser(description="Pyccel's command line interface.",
                      add_help=False)

    # ... Positional arguments
    group = parser.add_argument_group('Positional arguments')
    group.add_argument('filename', metavar='FILE', type=pathlib.Path,
                        help='Path (relative or absolute) to the Python file to be translated.')
    #...

    #... Help and Version
    import pyccel
    version = pyccel.__version__
    libpath = pyccel.__path__[0]
    python  = 'python {}.{}'.format(*sys.version_info)
    message = "pyccel {} from {} ({})".format(version, libpath, python)

    group = parser.add_argument_group('Basic options')
    group.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
    group.add_argument('-V', '--version', action='version', help='Show version and exit.', version=message)
    # ...

    # ... compiler syntax, semantic and codegen
    group = parser.add_argument_group('Pyccel compiling stages')
    group.add_argument('-x', '--syntax-only', action='store_true',
                       help='Stop Pyccel after syntactic parsing, before semantic analysis or code generation.')
    group.add_argument('-e', '--semantic-only', action='store_true',
                       help='Stop Pyccel after semantic analysis, before code generation.')
    group.add_argument('-t', '--convert-only', action='store_true',
                       help='Stop Pyccel after translation to the target language, before build.')
    # ...

    # ... backend compiler options
    group = parser.add_argument_group('Backend compiler options')

    group.add_argument('--language', choices=('fortran', 'c', 'python'), default='Fortran',
                       help='Target language for translation, i.e. the main language of the generated code (default: Fortran).',
                       type=str.lower)

    group.add_argument('--compiler', default = 'GNU',
                       help='Compiler family {GNU,intel,PGI,nvidia,LLVM} or JSON file containing a compiler description (default: GNU).')

    group.add_argument('--flags', type=str, \
                       help='Compiler flags.')
    group.add_argument('--wrapper-flags', type=str, \
                       help='Compiler flags for the wrapper.')
    group.add_argument('--debug', action=argparse.BooleanOptionalAction, default=None,
                        help='Compile the code with debug flags, or not.\n' \
                        ' Overrides the environment variable PYCCEL_DEBUG_MODE, if it exists. Otherwise default is False.')

    group.add_argument('--include',
                        type=str,
                        nargs='*',
                        dest='include',
                        default=(),
                        help='List of additional include directories.')

    group.add_argument('--libdir',
                        type=str,
                        nargs='*',
                        dest='libdir',
                        default=(),
                        help='List of additional library directories.')

    group.add_argument('--libs',
                        type=str,
                        nargs='*',
                        dest='libs',
                        default=(),
                        help='List of additional libraries to link with.')

    group.add_argument('--output', type=pathlib.Path, default = None,\
                       help="Folder in which the output is stored (default: FILE's folder).")
    # ...

    # ... Accelerators
    group = parser.add_argument_group('Accelerators options')
    group.add_argument('--mpi', action='store_true', \
                       help='Use MPI.')
    group.add_argument('--openmp', action='store_true', \
                       help='Use OpenMP.')
#    group.add_argument('--openacc', action='store_true', \
#                       help='Use OpenACC.') # [YG 17.06.2025] OpenACC is not supported yet
    # ...

    # ... Other options
    group = parser.add_argument_group('Other options')
    group.add_argument('-v', '--verbose', action='count', default = 0,\
                        help='Increase output verbosity (use -v, -vv, -vvv for more detailed output).')
    group.add_argument('--time-execution', action='store_true', \
                        help='Print the time spent in each section of the execution.')
    group.add_argument('--developer-mode', action='store_true', \
                        help='Show internal messages.')
    group.add_argument('--export-compile-info', metavar='COMPILER.json', type=pathlib.Path, default = None, \
                        help='Export all compiler information to a JSON file with the given path (relative or absolute).')
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
    from pyccel.codegen.pipeline  import execute_pyccel

    # ...
    filename = args.filename
    compiler = args.compiler
    mpi      = args.mpi
    openmp   = args.openmp
    openacc  = False  # [YG 17.06.2025] OpenACC is not supported yet
    output   = args.output or filename.parent

    if not args.conda_warnings:
        args.conda_warnings = 'basic'

    if args.convert_only or args.syntax_only or args.semantic_only:
        compiler = None
    # ...

    # ...
    compiler_export_file = args.export_compile_info

    # ... report error
    if filename.is_file():
        fext = filename.suffix
        if fext not in ['.py', '.pyi']:
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

    if compiler_export_file is not None:
        cext = compiler_export_file.suffix
        if cext == '':
            compiler_export_file = compiler_export_file.with_suffix('.json')
        elif cext != '.json':
            errors = Errors()
            # severity is error to avoid needing to catch exception
            errors.report('Wrong file extension. Expecting `json`, but found',
                          symbol=cext,
                          severity='error')
            errors.check()
            sys.exit(1)

    accelerators = []
    if mpi:
        accelerators.append("mpi")
    if openmp:
        accelerators.append("openmp")
    if openacc:
        accelerators.append("openacc")

    # ...

    # ...
    # this will initialize the singleton ErrorsMode
    # making this settings available everywhere
    err_mode = ErrorsMode()
    if args.developer_mode:
        err_mode.set_mode('developer')
    else:
        err_mode.set_mode(os.environ.get('PYCCEL_ERROR_MODE', 'user'))
    # ...

    base_dirpath = os.getcwd()

    if args.language == 'python' and args.output == '':
        print("Cannot output Python file to same folder as this would overwrite the original file. Please specify --output")
        sys.exit(1)

    try:
        # TODO: prune options
        execute_pyccel(str(filename),
                       syntax_only     = args.syntax_only,
                       semantic_only   = args.semantic_only,
                       convert_only    = args.convert_only,
                       verbose         = args.verbose,
                       time_execution  = args.time_execution,
                       language        = args.language,
                       compiler_family = compiler,
                       flags           = args.flags,
                       wrapper_flags   = args.wrapper_flags,
                       include         = args.include,
                       libdir          = args.libdir,
                       modules         = (),
                       libs            = args.libs,
                       debug           = args.debug,
                       accelerators    = accelerators,
                       folder          = str(output),
                       compiler_export_file = compiler_export_file,
                       conda_warnings  = args.conda_warnings)
    except PyccelError:
        sys.exit(1)
    finally:
        os.chdir(base_dirpath)
