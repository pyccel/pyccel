# coding: utf-8

from importlib.machinery import ExtensionFileLoader
from collections         import OrderedDict
from types               import ModuleType, FunctionType

import inspect
import subprocess
import importlib
import sys
import os
import string
import random
import shutil
import glob

from pyccel.parser                  import Parser
from pyccel.parser.errors           import Errors, PyccelError
from pyccel.parser.syntax.headers   import parse
from pyccel.codegen                 import Codegen
from pyccel.codegen.utilities       import execute_pyccel
from pyccel.codegen.utilities       import construct_flags as construct_flags_pyccel
from pyccel.ast                     import FunctionHeader
from pyccel.ast.utilities           import build_types_decorator
from pyccel.ast.core                import FunctionDef
from pyccel.ast.core                import Import
from pyccel.ast.core                import Module
from pyccel.ast.f2py                import F2PY_FunctionInterface
from pyccel.ast.f2py                import as_static_function
from pyccel.ast.f2py                import as_static_function_call
from pyccel.codegen.printing.pycode import pycode
from pyccel.codegen.printing.fcode  import fcode
from pyccel.ast.utilities           import get_external_function_from_ast
from pyccel.ast.utilities           import get_function_from_ast


#==============================================================================

PY_VERSION = sys.version_info[0:2]

#==============================================================================

def random_string( n ):
    # we remove uppercase letters because of f2py
    chars    = string.ascii_lowercase + string.digits
    selector = random.SystemRandom()
    return ''.join( selector.choice( chars ) for _ in range( n ) )

#==============================================================================

def mkdir_p(folder):
    if os.path.isdir(folder):
        return
    os.makedirs(folder)

#==============================================================================
def touch(path):
    with open(path, 'a'):
        os.utime(path, None)

#==============================================================================

def write_code(filename, code, folder=None):
    if not folder:
        folder = os.getcwd()

    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise ValueError('{} folder does not exist'.format(folder))

    filename = os.path.basename( filename )
    filename = os.path.join(folder, filename)

    # TODO check if __init__.py exists
    # add __init__.py for imports
    init_fname = os.path.join(folder, '__init__.py')
    touch(init_fname)

    f = open(filename, 'w')
    for line in code:
        f.write(line)
    f.close()

    return filename

#==============================================================================

def get_source_function(func):
    if not callable(func):
        raise TypeError('Expecting a callable function')

    lines = inspect.getsourcelines(func)
    lines = lines[0]
    # remove indentation if the first line is indented
    a = lines[0]
    leading_spaces = len(a) - len(a.lstrip())
    code = ''
    for a in lines:
        if leading_spaces > 0:
            line = a[leading_spaces:]
        else:
            line = a
        code = '{code}{line}'.format(code=code, line=line)

    return code

#==============================================================================

def construct_flags(compiler, extra_args = '', accelerator = None):

    f90flags   = ''
    opt        = ''

    if accelerator:
        if accelerator == 'openmp':
            if compiler == 'gfortran':
                extra_args += ' -lgomp '
                f90flags   += ' -fopenmp '

            elif compiler == 'ifort':
                extra_args += ' -liomp5 '
                f90flags   += ' -openmp -nostandard-realloc-lhs '
                opt         = """ --opt='-xhost -0fast' """

    return extra_args, f90flags, opt

#==============================================================================

def compile_fortran(source, modulename, extra_args='',libs=[], compiler=None ,
                    mpi=False, openmp=False, includes = [], only = []):
    """use f2py to compile a source code. We ensure here that the f2py used is
    the right one with respect to the python/numpy version, which is not the
    case if we run directly the command line f2py ..."""

    args_pattern = """  -c {compilers} --f90flags='{f90flags}' {opt} {libs} -m {modulename} {filename} {extra_args} {includes} {only}"""

    compilers  = ''
    f90flags   = ''
    opt        = ''

    if compiler == 'gfortran':
        _compiler = 'gnu95'

    elif compiler == 'ifort':
        _compiler = 'intelem'

    elif compiler == 'pgfortran':
       _compiler = 'pg'

    else:
        raise NotImplementedError('Only gfortran, ifort and pgi are available for the moment')

    extra_args, f90flags, opt = construct_flags( compiler,
                                                 extra_args = extra_args,
                                                 openmp = openmp )

    if mpi:
        compilers = '--f90exec=mpif90 '


    if compiler:
        compilers = compilers + '--fcompiler={}'.format(_compiler)

    if only:
        only = 'only: ' + ','.join(str(i) for i in only)
    else:
        only = ''

    if not libs:
        libs = ''

    if not includes:
        includes = ''

    try:
        filename = '{}.f90'.format( modulename.replace('.','/') )
        filename = os.path.basename( filename )
        f = open(filename, "w")
        for line in source:
            f.write(line)
        f.close()
        libs = ' '.join('-l'+i.lower() for i in libs)
        args = args_pattern.format( compilers  = compilers,
                                    f90flags   = f90flags,
                                    opt        = opt,
                                    libs       = libs,
                                    modulename = modulename.rpartition('.')[2],
                                    filename   = filename,
                                    extra_args = extra_args,
                                    includes   = includes,
                                    only       = only )
        
        cmd = """python{}.{} -m numpy.f2py {}"""
        
        
        cmd = cmd.format(PY_VERSION[0], PY_VERSION[1], args)

        output = subprocess.check_output(cmd, shell=True)
        return output, cmd

    finally:
        f.close()

#==============================================================================

# assumes relative path
# TODO add openacc
def compile_f2py( filename,
                  modulename=None,
                  extra_args='',
                  libs=[],
                  libdirs=[],
                  compiler=None ,
                  mpi=False,
                  accelerator=None,
                  includes = [],
                  only = [],
                  pyf = '' ):

    args_pattern = """  -c {compilers} --f90flags='{f90flags}' {opt} {libs} -m {modulename} {pyf} {filename} {libdirs} {extra_args} {includes} {only}"""

    compilers  = ''
    f90flags   = ''
    

    if compiler == 'gfortran':
        _compiler = 'gnu95'

    elif compiler == 'ifort':
        _compiler = 'intelem'

    elif compiler == 'pgfortran':
       _compiler = 'pg'
    
    else:
        raise NotImplementedError('Only gfortran ifort and pgi are available for the moment')

    if mpi:
        compilers = '--f90exec=mpif90 '

    if compiler:
        compilers = compilers + '--fcompiler={}'.format(_compiler)

    extra_args, f90flags, opt = construct_flags( compiler,
                                                 extra_args = extra_args,
                                                 accelerator = accelerator )
                                                 
    opt = "--opt='-O3'"

    if only:
        only = 'only: ' + ','.join(str(i) for i in only)
    else:
        only = ''

    if not libs:
        libs = ''

    if not libdirs:
        libdirs = ''

    if not includes:
        includes = ''

    if not modulename:
        modulename = filename.split('.')[0]

    libs = ' '.join('-l'+i.lower() for i in libs)
    libdirs = ' '.join('-L'+i for i in libdirs)

    args = args_pattern.format( compilers  = compilers,
                                f90flags   = f90flags,
                                opt        = opt,
                                libs       = libs,
                                libdirs    = libdirs,
                                modulename = modulename.rpartition('.')[2],
                                filename   = filename,
                                extra_args = extra_args,
                                includes   = includes,
                                only       = only,
                                pyf        = pyf )

    cmd = """python{}.{} -m numpy.f2py {}"""
    cmd = cmd.format(PY_VERSION[0], PY_VERSION[1], args)

    output = subprocess.check_output(cmd, shell=True)

#    # .... TODO: TO REMOVE
#    pattern_1 = 'f2py  {modulename}.f90 -h {modulename}.pyf -m {modulename}'
#    cmd_1 = pattern_1.format(modulename=modulename)
#
#    pattern_2 = 'f2py -c --fcompiler=gnu95 --f90flags=''  {modulename}.pyf {modulename}.f90 {libdirs} {libs}'
#    cmd_2 = pattern_2.format(modulename=modulename, libs=libs, libdirs=libdirs)
#
#    print('*****************')
#    print(cmd_1)
#    print(cmd_2)
#    print('*****************')
#    # ....

    return output, cmd

#==============================================================================
# TODO: move to 'pyccel.codegen.utilities', and use also in 'pyccel' command
def pyccelize_module(fname, *,
                     compiler    = None,
                     fflags      = None,
                     include     = [],
                     libdir      = [],
                     modules     = [],
                     libs        = [],
                     debug       = False,
                     verbose     = False,
                     extra_args  = '',
                     accelerator = None,
                     mpi         = False,
                     folder      = None):

    #------------------------------------------------------
    # NOTE:
    # [..]_dirname is the name of a directory
    # [..]_dirpath is the full (absolute) path of a directory
    #------------------------------------------------------

    # Store current directory
    base_dirpath = os.getcwd()

    pymod_filepath = os.path.abspath(fname)
    pymod_dirpath, pymod_filename = os.path.split(pymod_filepath)

    # Extract module name
    module_name = os.path.splitext(pymod_filename)[0]
    binary = '{}.o'.format(module_name)

    # Define working directory 'folder'
    if folder is None:
        folder = pymod_dirpath

    # Define directory name and path for pyccel & f2py build
    pyccel_dirname = '__pyccel__'
    pyccel_dirpath = os.path.join(folder, pyccel_dirname)

    # Create new directories if not existing
    mkdir_p(folder)
    mkdir_p(pyccel_dirpath)

    # Change working directory to 'folder'
    os.chdir(folder)

    # Choose Fortran compiler
    if compiler is None:
        if mpi == True:
            compiler = 'mpif90'
        else:
            compiler = 'gfortran'

    # ...
    # Construct flags for the Fortran compiler
    if fflags is None:
        fflags = construct_flags_pyccel( compiler,
                                         fflags=None,
                                         debug=debug,
                                         accelerator=accelerator,
                                         include=[],
                                         libdir=[] )

    # Build position-independent code, suited for use in shared library
    fflags = ' {} -fPIC '.format(fflags)
    # ...

    # Convert python to fortran using pyccel;
    # we ask for the AST so that we can get the FunctionDef node
    output, cmd, ast = execute_pyccel( pymod_filepath,
                                       compiler    = compiler,
                                       fflags      = fflags,
                                       debug       = debug,
                                       verbose     = verbose,
                                       accelerator = accelerator,
                                       include     = include,
                                       libdir      = libdir,
                                       modules     = modules,
                                       libs        = libs,
                                       binary      = None,
                                       output      = pyccel_dirpath,
                                       return_ast  = True )
    # ...

    # Change working directory to '__pyccel__'
    os.chdir(pyccel_dirpath)

    # ... construct a f2py interface for the assembly
    # be careful: because of f2py we must use lower case
    funcs = ast.routines + ast.interfaces

    # NOTE: we create an f2py interface for ALL functions
    f2py_filename = 'f2py_{}.f90'.format(module_name.lower())

    sharedlib_modname = module_name.lower()

    f2py_funcs = []
    for f in funcs:
        static_func = as_static_function_call(f, module_name, name=f.name)
        f2py_funcs.append(static_func)

    f2py_code = '\n\n'.join([fcode(f, ast.parser) for f in f2py_funcs])

    # Write file f2py_MOD.f90
    write_code(f2py_filename, f2py_code, folder=pyccel_dirpath)
    # ...

    # Create MOD.so shared library
    extra_args  = ' '.join([extra_args, '--no-wrap-functions', '--build-dir f2py_build'])
    output, cmd = compile_f2py( f2py_filename,
                                modulename  = sharedlib_modname,
                                libs        = [],
                                libdirs     = [],
                                includes    = binary,  # TODO: this is not an include...
                                extra_args  = extra_args,
                                compiler    = compiler,
                                accelerator = accelerator,
                                mpi         = mpi )

    # Obtain full name of shared library
    pattern = '{}*.so'.format(sharedlib_modname)
    sharedlib_filename = glob.glob(pattern)[0]

    # Move shared library to folder directory
    # (First construct absolute path of target location)
    sharedlib_filepath = os.path.join(folder, sharedlib_filename)
    shutil.move(sharedlib_filename, sharedlib_filepath)

    # Change working directory back to starting point
    os.chdir(base_dirpath)

    # Return absolute path of newly created shared library
    return sharedlib_filepath

#==============================================================================
def epyccel_seq(function_or_module,
                compiler    = None,
                fflags      = None,
                accelerator = None,
                verbose     = False,
                debug       = False,
                include     = [],
                libdir      = [],
                modules     = [],
                libs        = [],
                extra_args  = '',
                mpi         = False,
                folder      = None):

    # ... get the module source code
    if isinstance(function_or_module, FunctionType):
        func = function_or_module
        code = get_source_function(func)
        tag = random_string(8)
        module_name = 'mod_{}'.format(tag)
        fname       = '{}.py'.format(module_name)

    elif isinstance(function_or_module, ModuleType):
        module = function_or_module
        lines = inspect.getsourcelines(module)[0]
        code = ''.join(lines)
        module_name = module.__name__.split('.')[-1]
        fname       = module.__file__

    else:
        raise TypeError('> Expecting a FunctionType or a ModuleType')
    # ...

    pymod_filepath = os.path.abspath(fname)
    pymod_dirpath, pymod_filename = os.path.split(pymod_filepath)

    # Store current directory
    base_dirpath = os.getcwd()

    # Define working directory 'folder'
    if folder is None:
        folder = pymod_dirpath

    # Define directory name and path for epyccel files
    epyccel_dirname = '__epyccel__'
    epyccel_dirpath = os.path.join(folder, epyccel_dirname)

    # Create new directories if not existing
    mkdir_p(folder)
    mkdir_p(epyccel_dirpath)

    # Change working directory to '__epyccel__'
    os.chdir(epyccel_dirpath)

    # Store python file in '__epyccel__' folder, so that execute_pyccel can run
    fname = os.path.basename(fname)
    write_code(fname, code)

    # Generate shared library
    sharedlib_filepath = pyccelize_module(fname,
                                          compiler    = compiler,
                                          fflags      = fflags,
                                          include     = include,
                                          libdir      = libdir,
                                          modules     = modules,
                                          libs        = libs,
                                          debug       = debug,
                                          verbose     = verbose,
                                          extra_args  = extra_args,
                                          accelerator = accelerator,
                                          mpi         = mpi)

    if verbose:
        print( '> epyccel shared library has been created: {}'.format(sharedlib_filepath))

    # Create __init__.py file in epyccel directory
    with open('__init__.py', 'a') as f:
        pass

    # Change working directory to 'folder'
    os.chdir(folder)

    # Import shared library
    sys.path.append(epyccel_dirpath)
    package = importlib.import_module(epyccel_dirname + '.' + module_name)
    sys.path.remove(epyccel_dirpath)

    # Change working directory back to starting point
    os.chdir(base_dirpath)

    # Function case:
    if isinstance(function_or_module, FunctionType):
        return getattr(package, func.__name__.lower())

    # Module case:
    return package

#==============================================================================

def epyccel( inputs, **kwargs ):

    comm = kwargs.pop('comm', None)
    root = kwargs.pop('root', 0)
    bcast = kwargs.pop('bcast', True)

    if comm is not None:
        # TODO not tested for a function
        from mpi4py import MPI

        assert isinstance( comm, MPI.Comm )
        assert isinstance( root, int      )

        # Master process calls epyccel
        if comm.rank == root:
            kwargs['mpi'] = True

            fmod      = epyccel_seq( inputs, **kwargs )
            fmod_path = os.path.abspath(fmod.__file__)
            fmod_name = fmod.__name__

        else:
            fmod_path = None
            fmod_name = None
            fmod      = None

        if bcast:

            # Broadcast Fortran module path/name to all processes
            fmod_path = comm.bcast( fmod_path, root=root )
            fmod_name = comm.bcast( fmod_name, root=root )

            # Non-master processes import Fortran module directly from its path
            if comm.rank != root:
                folder = os.path.abspath(os.path.join(fmod_path, os.pardir))

                sys.path.append(folder)
                fmod = importlib.import_module( fmod_name )
                sys.path.remove(folder)

        # Return Fortran module
        return fmod

    else:
        return epyccel_seq( inputs, **kwargs )
