# coding: utf-8
#!/usr/bin/env python

# TODO add version
#  --version  show program's version number and exit

import sys
import os
import argparse

from collections  import OrderedDict

from pyccel.parser.errors     import Errors
from pyccel.parser.errors     import ErrorsMode
from pyccel.parser.messages   import INVALID_FILE_DIRECTORY, INVALID_FILE_EXTENSION
from pyccel.parser.utilities  import is_valid_filename_pyh, is_valid_filename_py
from pyccel.codegen.utilities import construct_flags
from pyccel.codegen.utilities import compile_fortran
from pyccel.codegen.utilities import execute_pyccel
from pyccel.ast.core          import Import
from pyccel.ast.core          import Module
from pyccel.ast.f2py          import as_static_function
from pyccel.ast.f2py          import as_static_function_call
from pyccel.ast.utilities     import get_external_function_from_ast

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

# TODO - remove output_dir froms args
#      - remove files from args
#      but quickstart and build are still calling it for the moment
def pyccel(files=None, openmp=None, openacc=None, output_dir=None, compiler='gfortran'):
    """
    pyccel console command.
    """
    parser = MyParser(description='pyccel command line')

    parser.add_argument('files', metavar='N', type=str, nargs='+',
                        help='a Pyccel file')

    # ... compiler syntax, semantic and codegen
    group = parser.add_argument_group('Pyccel compiling stages')
    group.add_argument('-x', '--syntax-only', action='store_true',
                       help='Using pyccel for Syntax Checking')
    group.add_argument('-e', '--semantic-only', action='store_true',
                       help='Using pyccel for Semantic Checking')
    group.add_argument('-t', '--convert-only', action='store_true',
                       help='Converts pyccel files only without build')
    group.add_argument('-f', '--f2py-compatible', action='store_true',
                        help='Converts pyccel files to be compiled by f2py')

    # ...

    # ... backend compiler options
    group = parser.add_argument_group('Backend compiler options')
    group.add_argument('--compiler', type=str, \
                       help='Used compiler')
    group.add_argument('--fflags', type=str, \
                       help='Fortran compiler flags.')
    group.add_argument('--debug', action='store_true', \
                       help='compiles the code in a debug mode.')
    group.add_argument('--include', type=str, \
                       help='path to include directory.')
    group.add_argument('--libdir', type=str, \
                       help='path to lib directory.')
    group.add_argument('--libs', type=str, \
                       help='list of libraries to link with.')
    group.add_argument('--output', type=str, default = '',\
                       help='folder in which the output is stored.')
    group.add_argument('--prefix', type=str, default = '',\
                       help='add prefix to the generated file.')
    group.add_argument('--prefix-module', type=str, default = '',\
                       help='add prefix module name.')

    group.add_argument('--language', type=str, help='target language')

    # ...

    # ... Accelerators
    group = parser.add_argument_group('Accelerators options')
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
    # ...

    # TODO move to another cmd line
    parser.add_argument('--analysis', action='store_true', \
                        help='enables code analysis mode.')
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

    debug   = args.debug
    verbose = args.verbose
    include = args.include
    fflags  = args.fflags
    libdir  = args.libdir
    libs    = args.libs
    output_folder = args.output
    prefix = args.prefix
    prefix_module = args.prefix_module
    language = args.language

    if (len(output_folder)>0 and output_folder[-1]!='/'):
        output_folder+='/'

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
        if args.language:
            settings['language'] = args.language
        ast = pyccel.annotate(**settings)
        name = os.path.basename(filename)
        name = os.path.splitext(name)[0]
        codegen = Codegen(ast, name)
        settings['prefix_module'] = prefix_module
        code = codegen.doprint(**settings)
        if prefix:
            name = '{prefix}{name}'.format(prefix=prefix, name=name)

        codegen.export(output_folder+name)

        for son in pyccel.sons:
            if 'print' in son.metavars.keys():
                name = son.filename.split('/')[-1].strip('.py')
                name = 'mod_'+name
                codegen = Codegen(son.ast, name)
                code = codegen.doprint()
                codegen.export()

    elif args.f2py_compatible:
        pyccel   = Parser(filename)
        ast      = pyccel.parse()
        settings = {}

        ast  = pyccel.annotate(**settings)
        name = os.path.basename(filename)
        name = os.path.splitext(name)[0]

        funcs     = ast.namespace.functions.values()
        namespace = ast.namespace.functions

        funcs, others = get_external_function_from_ast(funcs)
        static_funcs  = []
        imports       = []
        parents       = OrderedDict()

        for f in funcs:
            if f.is_external:
                static_func = as_static_function(f, str(f.name))
                namespace[str(f.name)] = static_func

            elif f.is_external_call:
                static_func = as_static_function_call(f, str(f.name))
                namespace[str(f.name)] = static_func
                imports += [Import(f.name, name)]

            static_funcs.append(static_func)
            parents[f.name] = f.name

        for f in others:
            imports += [Import(f.name, name)]

        imports += list(ast.namespace.imports['functions'])

        ast._ast = [Module( name,
                      variables = [],
                      funcs = static_funcs,
                      interfaces = [],
                      classes = [],
                      imports = imports )]

        codegen = Codegen(ast, name)

        settings['prefix_module'] = prefix_module
        codegen.doprint(**settings)
        if prefix:
            name = '{prefix}{name}'.format(prefix=prefix, name=name)

        codegen.export(output_folder+name)

    elif args.analysis:
        # TODO move to another cmd line
        from pyccel.complexity.arithmetic import OpComplexity
        complexity = OpComplexity(filename)
        print(" arithmetic cost         ~ " + str(complexity.cost()))

    else:
        # TODO shall we add them in the cmd line?
        modules = []
        binary = None

        execute_pyccel(filename,
                       compiler=compiler,
                       fflags=fflags,
                       debug=False,
                       verbose=verbose,
                       accelerator=accelerator,
                       include=include,
                       libdir=libdir,
                       modules=modules,
                       libs=libs,
                       binary=binary,
                       output=output_folder)



