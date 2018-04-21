# coding: utf-8
#!/usr/bin/env python

import sys
import os
import argparse

# TODO add version
#  --version  show program's version number and exit

from pyccel.parser.errors import Errors
from pyccel.parser.errors import ErrorsMode
from pyccel.parser.messages import INVALID_FILE_DIRECTORY, INVALID_FILE_EXTENSION
from pyccel.parser.utilities import is_valid_filename_pyh, is_valid_filename_py
from pyccel.codegen.utilities import construct_flags
from pyccel.codegen.utilities import compile_fortran
from pyccel.codegen.utilities import execute_pyccel

class MyParser(argparse.ArgumentParser):
    """
    Custom argument parser for printing help message in case of an error.
    See http://stackoverflow.com/questions/4042452/display-help-message-with-python-argparse-when-script-is-called-without-any-argu
    """
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

def _which(program):
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None

def pyccel(files=None, openmp=None, openacc=None, output_dir=None, compiler='gfortran'):
    """
    pyccel console command.
    """
    parser = MyParser(description='pyccel command line')

    # ...
    parser.add_argument('-x', '--syntax-only', action='store_true',
                        help='Using pyccel for Syntax Checking')
    parser.add_argument('-e', '--semantic-only', action='store_true',
                        help='Using pyccel for Semantic Checking')
    parser.add_argument('-t', '--convert-only', action='store_true',
                        help='Converts pyccel files only without build')


    parser.add_argument('--compiler', type=str, \
                        help='Used compiler')
    parser.add_argument('--openmp', action='store_true', \
                        help='uses openmp')
    parser.add_argument('--openacc', action='store_true', \
                        help='uses openacc')
    parser.add_argument('--debug', action='store_true', \
                        help='compiles the code in a debug mode.')


    parser.add_argument('--verbose', action='store_true', \
                        help='enables verbose mode.')
    parser.add_argument('--include', type=str, \
                        help='path to include directory.')
    parser.add_argument('--libdir', type=str, \
                        help='path to lib directory.')
    parser.add_argument('--libs', type=str, \
                        help='list of libraries to link with.')


    parser.add_argument('--developer-mode', action='store_true', \
                        help='shows internal messages if True')
    parser.add_argument('--output-dir', type=str, \
                        help='Output directory.')

    # TODO move to another cmd line
#    parser.add_argument('--analysis', action='store_true', \
#                        help='enables code analysis mode.')


    if not files:
        parser.add_argument('files', metavar='N', type=str, nargs='+',
                            help='a Pyccel file')
    # ...

    # ...
    args = parser.parse_args()
    # ...

    # ...
    if not files:
        files = args.files

    if args.compiler:
        compiler = args.compiler

    if not openmp:
        openmp = args.openmp

    if not openacc:
        openacc = args.openacc

    if not output_dir:
        output_dir = args.output_dir

    if args.convert_only or args.syntax_only or args.semantic_only:
        compiler = None
    # ...

    # ...
    if not files:
        raise ValueError("a python filename must be provided.")

    if len(files) > 1:
        raise ValueError('Expecting one single file for the moment.')
    # ...

    filename = files[0]

    # ... report error
    if os.path.isfile(filename):
        # we don't use is_valid_filename_py since it uses absolute path
        # file extension
        ext = filename.split('.')[-1]
        if not(ext in ['py', 'pyh']):
            errors = Errors()
            errors.report(INVALID_FILE_EXTENSION,
                          symbol=ext,
                          severity='fatal')
            errors.check()
            raise SystemExit(0)
    else:
        # we use Pyccel error manager, although we can do it in other ways
        errors = Errors()
        errors.report(INVALID_FILE_DIRECTORY,
                      symbol=filename,
                      severity='fatal')
        errors.check()
        raise SystemExit(0)
    # ...

    if compiler:
        if _which(compiler) is None:
            raise ValueError('Could not find {0}'.format(compiler))

    accelerator = None
    if openmp:
        accelerator = "openmp"
    if openacc:
        accelerator = "openacc"

    debug    = args.debug
    verbose  = args.verbose
    include  = args.include
    libdir   = args.libdir
    libs     = args.libs

    if not include:
        include = []
    if not libdir:
        libdir = []
    if not libs:
        libs = []
    # ...

    # ...
    if args.developer_mode:
        # this will initialize the singelton ErrorsMode
        # making this settings available everywhere
        err_mode = ErrorsMode()
        err_mode.set_mode('developer')
    # ...

    # ...
    from pyccel.parser import Parser
    from pyccel.codegen import Codegen

    if args.syntax_only:
        pyccel = Parser(filename)
        ast = pyccel.parse()

    elif args.semantic_only:
        pyccel = Parser(filename)
        ast = pyccel.parse()

        settings = {}
        ast = pyccel.annotate(**settings)

    elif args.convert_only:
        pyccel = Parser(filename)
        ast = pyccel.parse()

        settings = {}
        ast = pyccel.annotate(**settings)

        name = os.path.basename(filename)
        name = os.path.splitext(name)[0]
        codegen = Codegen(ast, name)
        code = codegen.doprint()
        codegen.export()

    else:
        # TODO shall we add them in the cmd line?
        modules = []
        binary = None

        execute_pyccel(filename,
                       compiler=compiler,
                       debug=False,
                       accelerator=accelerator,
                       include=include,
                       libdir=libdir,
                       modules=modules,
                       libs=libs,
                       binary=binary)

#    elif analysis:
#        # TODO move to another cmd line
#
#        from pyccel.complexity.memory import MemComplexity
#
#        local_vars = []
#        if args.local_vars:
#            local_vars = args.local_vars.split(',')
#        complexity = MemComplexity(filename)
#        complexity.intensity(verbose=True, local_vars=local_vars)
    # ...
