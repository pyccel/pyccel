#!/usr/bin/env python
# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

import sys
import os
import argparse

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
# TODO - remove output_dir froms args
#      - remove files from args
#      but quickstart and build are still calling it for the moment
def pyccel(files=None, mpi=None, openmp=None, openacc=None, output_dir=None, compiler=None):
    """
    pyccel console command.
    """
    parser = MyParser(description='pyccel command line')

    parser.add_argument('files', metavar='N', type=str, nargs='*',
                        help='a Pyccel file')

    #... Version
    import pyccel
    version = pyccel.__version__
    libpath = pyccel.__path__[0]
    python  = 'python {}.{}'.format(*sys.version_info)
    message = "pyccel {} from {} ({})".format(version, libpath, python)
    parser.add_argument('-V', '--version', action='version', version=message)
    # ...

    # ... compiler syntax, semantic and codegen
    group = parser.add_argument_group('Pyccel compiling stages')
    group.add_argument('-x', '--syntax-only', action='store_true',
                       help='Using pyccel for Syntax Checking')
    group.add_argument('-e', '--semantic-only', action='store_true',
                       help='Using pyccel for Semantic Checking')
    group.add_argument('-t', '--convert-only', action='store_true',
                       help='Converts pyccel files only without build')
#    group.add_argument('-f', '--f2py-compatible', action='store_true',
#                        help='Converts pyccel files to be compiled by f2py')

    # ...

    # ... backend compiler options
    group = parser.add_argument_group('Backend compiler options')

    group.add_argument('--language', choices=('fortran', 'c', 'python'), help='Generated language')

    group.add_argument('--compiler', help='Compiler family or json file containing a compiler description {GNU,intel,PGI}')

    group.add_argument('--flags', type=str, \
                       help='Compiler flags.')
    group.add_argument('--wrapper-flags', type=str, \
                       help='Compiler flags for the wrapper.')
    group.add_argument('--debug', action='store_true', \
                       help='compiles the code in a debug mode.')

    group.add_argument('--include',
                        type=str,
                        nargs='*',
                        dest='includes',
                        default=(),
                        help='list of include directories.')

    group.add_argument('--libdir',
                        type=str,
                        nargs='*',
                        dest='libdirs',
                        default=(),
                        help='list of library directories.')

    group.add_argument('--libs',
                        type=str,
                        nargs='*',
                        dest='libs',
                        default=(),
                        help='list of libraries to link with.')

    group.add_argument('--output', type=str, default = '',\
                       help='folder in which the output is stored.')

    # ...

    # ... Accelerators
    group = parser.add_argument_group('Accelerators options')
    group.add_argument('--mpi', action='store_true', \
                       help='uses mpi')
    group.add_argument('--openmp', action='store_true', \
                       help='uses openmp')
    group.add_argument('--openacc', action='store_true', \
                       help='uses openacc')
    # ...

    # ... Other options
    group = parser.add_argument_group('Other options')
    group.add_argument('--verbose', action='store_true', \
                        help='enables verbose mode.')
    group.add_argument('--developer-mode', action='store_true', \
                        help='shows internal messages')
    group.add_argument('--export-compile-info', type=str, default = None, \
                        help='file to which the compiler json file is exported')
    group.add_argument('--disable-conda-warnings', action='store_true', \
                         help='disable warnings when pyccel running in conda environment')
    group.add_argument('--detailed-conda-warnings', action='store_true', \
                         help='show list of conda paths ignored by pyccel')
    # ...

    # TODO move to another cmd line
    parser.add_argument('--analysis', action='store_true', \
                        help='enables code analysis mode.')
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
    if not files:
        files = args.files

    if args.compiler:
        compiler = args.compiler

    if not mpi:
        mpi = args.mpi

    if not openmp:
        openmp = args.openmp

    if not openacc:
        openacc = args.openacc

    if args.convert_only or args.syntax_only or args.semantic_only:
        compiler = None
    # ...

    # ...

    if len(files) > 1:
        errors = Errors()
        # severity is error to avoid needing to catch exception
        errors.report('Pyccel can currently only handle 1 file at a time',
                      severity='error')
        errors.check()
        sys.exit(1)
    # ...

    compiler_export_file = args.export_compile_info

    if len(files) == 0:
        if compiler_export_file is None:
            parser.error("please specify a file to pyccelise")
        else:
            filename = ''
    else:
        filename = files[0]

        # ... report error
        if os.path.isfile(filename):
            # we don't use is_valid_filename_py since it uses absolute path
            # file extension
            ext = filename.split('.')[-1]
            if not(ext in ['py', 'pyh']):
                errors = Errors()
                # severity is error to avoid needing to catch exception
                errors.report(INVALID_FILE_EXTENSION,
                              symbol=ext,
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
        _, ext = os.path.splitext(compiler_export_file)
        if ext == '':
            compiler_export_file = compiler_export_file + '.json'
        elif ext != '.json':
            errors = Errors()
            # severity is error to avoid needing to catch exception
            errors.report('Wrong file extension. Expecting `json`, but found',
                          symbol=ext,
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
    if args.developer_mode:
        # this will initialize the singelton ErrorsMode
        # making this settings available everywhere
        err_mode = ErrorsMode()
        err_mode.set_mode('developer')
    # ...

    base_dirpath = os.getcwd()

    if args.language == 'python' and args.output == '':
        print("Cannot output python file to same folder as this would overwrite the original file. Please specify --output")
        sys.exit(1)

    try:
        # TODO: prune options
        execute_pyccel(filename,
                       syntax_only   = args.syntax_only,
                       semantic_only = args.semantic_only,
                       convert_only  = args.convert_only,
                       verbose       = args.verbose,
                       language      = args.language,
                       compiler      = compiler,
                       fflags        = args.flags,
                       wrapper_flags = args.wrapper_flags,
                       includes      = args.includes,
                       libdirs       = args.libdirs,
                       modules       = (),
                       libs          = args.libs,
                       debug         = args.debug,
                       accelerators  = accelerators,
                       folder        = args.output,
                       compiler_export_file = compiler_export_file,
                       is_conda_warnings_disabled = args.disable_conda_warnings,
                       is_conda_warnings_detailed = args.detailed_conda_warnings)
    except PyccelError:
        sys.exit(1)
    finally:
        os.chdir(base_dirpath)

#==============================================================================
# NOTE: left here for later reference
#
#    group.add_argument('--prefix', type=str, default = '',\
#                       help='add prefix to the generated file.')
#    group.add_argument('--prefix-module', type=str, default = '',\
#                       help='add prefix module name.')
#    group.add_argument('--language', type=str, help='target language')
#
#    ...
#
#    prefix = args.prefix
#    prefix_module = args.prefix_module
#    language = args.language
#
#    output_folder = args.output
#
#    if (len(output_folder)>0 and output_folder[-1]!='/'):
#        output_folder+='/'
#
#    ...
#
#    elif args.convert_only:
#        pyccel = Parser(filename)
#        ast = pyccel.parse()
#        settings = {}
#        if args.language:
#            settings['language'] = args.language
#        ast = pyccel.annotate(**settings)
#        name = os.path.basename(filename)
#        name = os.path.splitext(name)[0]
#        codegen = Codegen(ast, name)
#        settings['prefix_module'] = prefix_module
#        code = codegen.doprint(**settings)
#        if prefix:
#            name = '{prefix}{name}'.format(prefix=prefix, name=name)
#
#        codegen.export(output_folder+name)
#
#        for son in pyccel.sons:
#            if 'print' in son.metavars.keys():
#                name = son.filename.split('/')[-1].strip('.py')
#                name = 'mod_'+name
#                codegen = Codegen(son.ast, name)
#                code = codegen.doprint()
#                codegen.export()
