# coding: utf-8

import sys
import subprocess
import os
import glob

from pyccel.ast.f2py                import as_static_function_call
from pyccel.codegen.printing.fcode  import fcode

__all__ = ['compile_f2py', 'create_shared_library']

#==============================================================================

PY_VERSION = sys.version_info[0:2]

#==============================================================================
# assumes relative path
# TODO add openacc
# TODO [YG, 04.02.2020] sanitize arguments to protect against shell injection
def compile_f2py( filename, *,
                  modulename=None,
                  extra_args='',
                  libs=(),
                  libdirs=(),
                  compiler=None,
                  mpi_compiler=None,
                  accelerator=None,
                  includes = '',
                  only = (),
                  pyf = '' ):

    args_pattern = """  -c {compilers} --f90flags='{f90flags}' {opt} {libs} -m {modulename} {pyf} {filename} {libdirs} {extra_args} {includes} {only}"""

    compilers  = ''
    f90flags   = ''

    #... Determine Fortran compiler vendor for F2PY
    if compiler == 'gfortran':
        _vendor = 'gnu95'

    elif compiler == 'ifort':
        _vendor = 'intelem'

    elif compiler == 'pgfortran':
       _vendor = 'pg'

    else:
        raise NotImplementedError('Only gfortran ifort and pgi are available for the moment')
    #...

    if mpi_compiler:
        compilers = '--f90exec={}'.format(mpi_compiler)

    if compiler:
        compilers = compilers + ' --fcompiler={}'.format(_vendor)

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

    libs = ' '.join('-l'+i.lower() for i in libs) # because of f2py we must use lower case
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
def create_shared_library(codegen,
                          pyccel_dirpath,
                          compiler,
                          mpi_compiler,
                          accelerator,
                          dep_mods,
                          extra_args='',
                          sharedlib_modname=None):

    # Consistency checks
    if not codegen.is_module:
        raise TypeError('Expected Module')

    # Get module name
    module_name = codegen.name

    # Change working directory to '__pyccel__'
    base_dirpath = os.getcwd()
    os.chdir(pyccel_dirpath)

    # Construct f2py interface for assembly and write it to file f2py_MOD.f90
    # be careful: because of f2py we must use lower case
    funcs = codegen.routines + codegen.interfaces
    f2py_funcs = [as_static_function_call(f, module_name, name=f.name) for f in funcs]
    f2py_code = '\n\n'.join([fcode(f, codegen.parser) for f in f2py_funcs])
    f2py_filename = 'f2py_{}.f90'.format(module_name)
    with open(f2py_filename, 'w') as f:
        f.writelines(f2py_code)

    object_files = ' '.join(['{}.o'.format(m) for m in dep_mods])
    # ...

    # Name of shared library
    if sharedlib_modname is None:
        sharedlib_modname = module_name

    # Create MOD.so shared library
    extra_args  = ' '.join([extra_args, '--no-wrap-functions', '--build-dir f2py_build'])
    compile_f2py(f2py_filename,
                 modulename  = sharedlib_modname,
                 libs        = (),
                 libdirs     = (),
                 includes    = object_files,  # TODO: this is not an include...
                 extra_args  = extra_args,
                 compiler    = compiler,
                 mpi_compiler= mpi_compiler,
                 accelerator = accelerator)

    # Obtain absolute path of newly created shared library
    pattern = '{}*.so'.format(sharedlib_modname)
    sharedlib_filename = glob.glob(pattern)[0]
    sharedlib_filepath = os.path.abspath(sharedlib_filename)

    # Change working directory back to starting point
    os.chdir(base_dirpath)

    # Return absolute path of shared library
    return sharedlib_filepath
