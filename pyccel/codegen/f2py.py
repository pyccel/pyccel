# coding: utf-8

import sys
import subprocess
import os
import glob
import shutil

from pyccel.codegen.utilities       import construct_flags
from pyccel.codegen.utilities       import execute_pyccel
from pyccel.codegen.printing.fcode  import fcode
from pyccel.ast.f2py                import as_static_function_call

__all__ = ['compile_f2py', 'pyccelize_module']

#==============================================================================

PY_VERSION = sys.version_info[0:2]

#==============================================================================
# assumes relative path
# TODO add openacc
def compile_f2py( filename,
                  modulename=None,
                  extra_args='',
                  libs=[],
                  libdirs=[],
                  compiler=None,
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

    f90flags = ''
    opt = "--opt='-O3'"

    if accelerator:
        if accelerator == 'openmp':
            if compiler == 'gfortran':
                extra_args += ' -lgomp '
                f90flags   += ' -fopenmp '

            elif compiler == 'ifort':
                extra_args += ' -liomp5 '
                f90flags   += ' -openmp -nostandard-realloc-lhs '
                opt         = """ --opt='-xhost -0fast' """

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

    # Define working directory 'folder'
    if folder is None:
        folder = pymod_dirpath

    # Define directory name and path for pyccel & f2py build
    pyccel_dirname = '__pyccel__'
    pyccel_dirpath = os.path.join(folder, pyccel_dirname)

    # Create new directories if not existing
    os.makedirs(folder, exist_ok=True)
    os.makedirs(pyccel_dirpath, exist_ok=True)

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
        fflags = construct_flags(compiler,
                                 fflags=None,
                                 debug=debug,
                                 accelerator=accelerator,
                                 include=[],
                                 libdir=[])

    # Build position-independent code, suited for use in shared library
    fflags = ' {} -fPIC '.format(fflags)
    # ...

    # Convert python to fortran using pyccel
    parser, codegen = execute_pyccel( pymod_filepath,
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
                                       output      = pyccel_dirpath)
    # ...
    # Determine all .o files needed by shared library
    def get_module_dependencies(parser, mods=[]):
        mods = mods + [os.path.splitext(os.path.basename(parser.filename))[0]]
        for son in parser.sons:
            mods = get_module_dependencies(son, mods)
        return mods

    dep_mods = get_module_dependencies(parser)
    binary = ' '.join(['{}.o'.format(m) for m in dep_mods])
    # ...

    # Change working directory to '__pyccel__'
    os.chdir(pyccel_dirpath)

    # ... construct a f2py interface for the assembly
    # be careful: because of f2py we must use lower case
    funcs = codegen.routines + codegen.interfaces

    # NOTE: we create an f2py interface for ALL functions
    f2py_filename = 'f2py_{}.f90'.format(module_name.lower())

    sharedlib_modname = module_name.lower()

    f2py_funcs = []
    for f in funcs:
        static_func = as_static_function_call(f, module_name, name=f.name)
        f2py_funcs.append(static_func)

    f2py_code = '\n\n'.join([fcode(f, codegen.parser) for f in f2py_funcs])

    # Write file f2py_MOD.f90
    with open(f2py_filename, 'w') as f:
        f.writelines(f2py_code)
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
