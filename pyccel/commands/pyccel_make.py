#!/usr/bin/env python
# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
File containing the pyccel-make command line interface.
"""

import argparse
import glob
import os
import sys
from pathlib import Path

def pyccel_make_command() -> None:
    """
    Pyccel console command.

    The command line interface allowing pyccel-make to be called.
    """
    parser = argparse.ArgumentParser(description="Pyccel's command line interface for multi-file projects.",
            add_help = False)

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
    group = parser.add_argument_group('File specification',
            description = "Use one of the below methods to specify which files should be translated."
            ).add_mutually_exclusive_group(required=True)
    group.add_argument('-f', '--files', nargs='+', type=Path, metavar='FILE',
            help="A list of files to be translated as a project.")
    group.add_argument('-g', '--glob', type=str,
            help=("A glob that should be used to recognise files to be translated as a project (e.g. '**/*.py'). "
                  "Note: quote the pattern to prevent shell expansion."))
    group.add_argument('-d', '--file-descr', type=Path,
            help="A UTF-8 text file containing the paths to the files to be translated as a project. One path (relative or absolute) per line.")

    # ... backend compiler options
    group = parser.add_argument_group('Backend selection')

    group.add_argument('--language', choices=('fortran', 'c', 'python'), default='Fortran',
                       help='Target language for translation, i.e. the main language of the generated code (default: Fortran).',
                       type=str.lower)
    group.add_argument('--build-system', choices=('meson', 'cmake'), default='meson',
                       help='Chosen build system for translation, i.e. the tool that will be used to compile the generated code (default: meson).',
                       type=str.lower)

    # ... Compiler options
    group = parser.add_argument_group('Compiler configuration (mutually exclusive options)')
    compiler_group = group.add_mutually_exclusive_group(required=False)
    compiler_group.add_argument('--compiler-family',
                                type=str,
                                default='GNU',
                                metavar='FAMILY',
                                help='Compiler family {GNU,intel,PGI,nvidia,LLVM} (default: GNU).')
    compiler_group.add_argument('--compiler-config',
                                type=Path,
                                default=None,
                                metavar='CONFIG.json',
                                help='Load all compiler information from a JSON file with the given path (relative or absolute).')

    # ... Additional compiler options
    group = parser.add_argument_group('Additional compiler options')
    group.add_argument('--debug', action=argparse.BooleanOptionalAction, default=None,
                        help='Compile the code with debug flags, or not.\n' \
                        ' Overrides the environment variable PYCCEL_DEBUG_MODE, if it exists. Otherwise default is False.')
    group.add_argument('--output', type=Path, default = None,\
                       help="Folder in which the output is stored (default: FILE's folder).")

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
    group.add_argument('--conda-warnings', choices=('off', 'basic', 'verbose'), default='basic',
                        help='Specify the level of Conda warnings to display (default: basic).')
    group.add_argument('-t', '--convert-only', action='store_true',
                       help='Stop Pyccel after translation to the target language, before build.')
    # ...
    # ...
    args = parser.parse_args()

    from pyccel.codegen.make_pipeline  import execute_pyccel_make
    from pyccel.errors.errors     import Errors, PyccelError
    from pyccel.errors.errors     import ErrorsMode

    errors = Errors()

    if args.files:
        files = args.files
    elif args.glob:
        files = [Path(f) for f in glob.glob(args.glob, recursive=True)]
    elif args.file_descr:
        with open(args.file_descr, 'r', encoding='utf-8') as f:
            files = [Path(fname.strip()) for fname in f.readlines()]
    else:
        raise NotImplementedError("No file specified")

    for f in files:
        if not f.exists():
            errors.report(f"File not found : {f}", severity='error')
        if f.suffix != '.py':
            errors.report(f"Expected Python file, received : {f}", severity='error')

    cwd = os.getcwd()
    files = [f.relative_to(cwd) if f.is_absolute() else f for f in files]

    errors.check()

    if errors.has_errors():
        sys.exit(1)

    accelerators = []
    if args.mpi:
        accelerators.append("mpi")
    if args.openmp:
        accelerators.append("openmp")
    #if args.openacc:
    #    accelerators.append("openacc")

    # ...
    # this will initialize the singleton ErrorsMode
    # making this settings available everywhere
    err_mode = ErrorsMode()
    if args.developer_mode:
        err_mode.set_mode('developer')
    else:
        err_mode.set_mode(os.environ.get('PYCCEL_ERROR_MODE', 'user'))
    # ...

    try:
        execute_pyccel_make(files,
                            verbose = args.verbose,
                            time_execution = args.time_execution,
                            folder = args.output,
                            language = args.language,
                            compiler_family = args.compiler_family,
                            build_system = args.build_system,
                            debug = args.debug,
                            accelerators = accelerators,
                            conda_warnings = args.conda_warnings,
                            build_code = not args.convert_only)
    except PyccelError:
        sys.exit(1)
