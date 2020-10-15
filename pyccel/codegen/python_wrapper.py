# coding: utf-8

import sys
import subprocess
import os
import glob
import warnings

from pyccel.ast.f2py                        import as_static_function_call
from pyccel.ast.core                        import SeparatorComment
from pyccel.codegen.printing.fcode          import fcode
from pyccel.codegen.printing.cwrappercode   import cwrappercode
from .cwrapper import create_c_setup

from pyccel.errors.errors import Errors

errors = Errors()

__all__ = ['compile_f2py', 'create_shared_library']

#==============================================================================

PY_VERSION = sys.version_info[0:2]

#==============================================================================
# assumes relative path
# TODO add openacc
# TODO [YG, 04.02.2020] sanitize arguments to protect against shell injection
def compile_f2py( filename, *,
                  language='fortran',
                  modulename=None,
                  extra_args='',
                  libs=(),
                  libdirs=(),
                  compiler=None,
                  mpi_compiler=None,
                  accelerator=None,
                  includes = '',
                  only = (),
                  pyf = '',
                  verbose = False ):

    args_pattern = """  -c {compilers} --f90flags="{f90flags}" {opt} {libs} -m {modulename} {pyf} {filename} {libdirs} {extra_args} {includes} {only} {verbose_str}"""

    compilers  = ''
    f90flags   = ''
    opt        = '--opt="-O3"'

    #... Determine Fortran compiler vendor for F2PY
    if compiler == 'gfortran':
        _vendor = 'gnu95'

    elif compiler == 'gcc':
        _vendor = 'unix'

    elif compiler in ['ifort', 'icc']:
        _vendor = 'intelem'

    elif compiler == 'pgfortran':
        _vendor = 'pg'

    else:
        raise NotImplementedError('Only gfortran, gcc, ifort, icc and pgi are available for the moment')
    #...

    if mpi_compiler:
        compilers = '--f90exec' if language == 'fortran' else '--compiler'
        if sys.platform == 'win32' and mpi_compiler == 'mpif90':
            compilers = compilers + '=gfortran'
        else:
            compilers = compilers + '={}'.format(mpi_compiler)

    if compiler:
        if language == 'fortran':
            compilers = compilers + ' --fcompiler={}'.format(_vendor)
        else:
            compilers = compilers + ' --compiler={}'.format(_vendor)


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

    if not verbose:
        verbose_str = '--quiet'
    else:
        verbose_str = '--verbose'

    args = args_pattern.format( compilers   = compilers,
                                f90flags    = f90flags,
                                opt         = opt,
                                libs        = libs,
                                libdirs     = libdirs,
                                modulename  = modulename.rpartition('.')[2],
                                filename    = filename,
                                extra_args  = extra_args,
                                includes    = includes,
                                only        = only,
                                pyf         = pyf,
                                verbose_str = verbose_str )

    cmd = """{} -m numpy.f2py {}"""
    cmd = cmd.format(sys.executable, args)

    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

    if verbose:
        print(cmd)

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
                          language,
                          pyccel_dirpath,
                          compiler,
                          mpi_compiler,
                          accelerator,
                          dep_mods,
                          libs,
                          libdirs,
                          includes='',
                          flags = '',
                          extra_args='',
                          sharedlib_modname=None,
                          verbose = False):

    # Consistency checks
    if not codegen.is_module:
        raise TypeError('Expected Module')

    # Get module name
    module_name = codegen.name

    # Change working directory to '__pyccel__'
    base_dirpath = os.getcwd()
    os.chdir(pyccel_dirpath)

    # Name of shared library
    if sharedlib_modname is None:
        sharedlib_modname = module_name

    sharedlib_folder = ''

    if language == 'c':
        module_old_name = codegen.expr.name
        codegen.expr.set_name(sharedlib_modname)
        wrapper_code = cwrappercode(codegen.expr, codegen.parser)
        codegen.expr.set_name(module_old_name)
        errors.check()
        errors.reset()
        wrapper_filename_root = '{}_wrapper'.format(module_name)
        wrapper_filename = '{}.c'.format(wrapper_filename_root)

        with open(wrapper_filename, 'w') as f:
            f.writelines(wrapper_code)

        dep_mods = (wrapper_filename_root, *dep_mods)
        setup_code = create_c_setup(sharedlib_modname, dep_mods,
                compiler, includes, libs, libdirs, flags)
        setup_filename = "setup_{}.py".format(module_name)

        with open(setup_filename, 'w') as f:
            f.writelines(setup_code)

        setup_filename = os.path.join(pyccel_dirpath, setup_filename)
        cmd = [sys.executable, setup_filename, "build"]

        if verbose:
            print(' '.join(cmd))
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        out, err = p.communicate()
        if verbose:
            print(out)
        if p.returncode != 0:
            err_msg = "Failed to build module"
            if verbose:
                err_msg += "\n" + err
            raise RuntimeError(err_msg)
        if err:
            warnings.warn(UserWarning(err))

        sharedlib_folder += 'build/lib*/'

    elif language == 'fortran':
        # Construct f2py interface for assembly and write it to file f2py_MOD.f90
        # be careful: because of f2py we must use lower case
        funcs = codegen.routines + codegen.interfaces
        sep = fcode(SeparatorComment(40), codegen.parser)
        f2py_funcs = [as_static_function_call(f, module_name, name=f.name) for f in funcs]
        f2py_code = '\n'.join([sep + fcode(f, codegen.parser) + sep for f in f2py_funcs])
        f2py_filename = 'f2py_{}.f90'.format(module_name)

        with open(f2py_filename, 'w') as f:
            f.writelines(f2py_code)

        object_files = ' '.join(['"{}.o"'.format(m) for m in dep_mods])


        # ...

        # Create MOD.so shared library
        extra_args  = ' '.join([extra_args, '--no-wrap-functions', '--build-dir f2py_build'])
        compile_f2py(f2py_filename,
                     language    = language,
                     modulename  = sharedlib_modname,
                     libs        = libs,
                     libdirs     = libdirs,
                     includes    = object_files,  # TODO: this is not an include...
                     extra_args  = extra_args,
                     compiler    = compiler,
                     mpi_compiler= mpi_compiler,
                     accelerator = accelerator,
                     verbose     = verbose )

    # Obtain absolute path of newly created shared library

    # Set file name extension of Python extension module
    if os.name == 'nt':  # Windows
        extext = 'pyd'
    else:
        extext = 'so'
    pattern = '{}{}*.{}'.format(sharedlib_folder, sharedlib_modname, extext)
    sharedlib_filename = glob.glob(pattern)[0]
    sharedlib_filepath = os.path.abspath(sharedlib_filename)

    # Change working directory back to starting point
    os.chdir(base_dirpath)

    # Return absolute path of shared library
    return sharedlib_filepath
