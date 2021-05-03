# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

import sys
import subprocess
import os
import glob
import warnings
from filelock import FileLock

from pyccel.ast.bind_c                      import as_static_module
from pyccel.ast.core                        import SeparatorComment
from pyccel.codegen.printing.fcode          import fcode
from pyccel.codegen.printing.cwrappercode   import cwrappercode
from pyccel.codegen.utilities               import compile_files, get_gfortran_library_dir
from .cwrapper import create_c_setup

from pyccel.errors.errors import Errors

errors = Errors()

__all__ = ['create_shared_library', 'fortran_c_flag_equivalence']

#==============================================================================

PY_VERSION = sys.version_info[0:2]

fortran_c_flag_equivalence = {'-Wconversion-extra' : '-Wconversion' }

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

    if language in ['c', 'fortran']:
        extra_libs = []
        extra_libdirs = []
        if language == 'fortran':
            # Construct static interface for passing array shapes and write it to file bind_c_MOD.f90
            new_module_name = 'bind_c_{}'.format(module_name)
            bind_c_mod = as_static_module(codegen.routines, module_name, new_module_name)
            bind_c_code = fcode(bind_c_mod, codegen.parser)
            bind_c_filename = '{}.f90'.format(new_module_name)

            with open(bind_c_filename, 'w') as f:
                f.writelines(bind_c_code)

            compile_files(bind_c_filename, compiler, flags,
                binary=None,
                verbose=verbose,
                is_module=True,
                output=pyccel_dirpath,
                libs=libs,
                libdirs=libdirs,
                language=language)

            dep_mods = (os.path.join(pyccel_dirpath,'bind_c_{}'.format(module_name)), *dep_mods)
            if compiler == 'gfortran':
                extra_libs.append('gfortran')
                extra_libdirs.append(get_gfortran_library_dir())
            elif compiler == 'ifort':
                extra_libs.append('ifcore')

        if sys.platform == 'win32':
            extra_libs.append('quadmath')

        module_old_name = codegen.expr.name
        codegen.expr.set_name(sharedlib_modname)
        wrapper_code = cwrappercode(codegen.expr, codegen.parser, language)
        if errors.has_errors():
            return

        codegen.expr.set_name(module_old_name)
        wrapper_filename_root = '{}_wrapper'.format(module_name)
        wrapper_filename = '{}.c'.format(wrapper_filename_root)

        with open(wrapper_filename, 'w') as f:
            f.writelines(wrapper_code)

        c_flags = [fortran_c_flag_equivalence[f] if f in fortran_c_flag_equivalence \
                else f for f in flags]

        if sys.platform == "darwin" and "-fopenmp" in c_flags and "-Xpreprocessor" not in c_flags:
            idx = 0
            while idx < len(c_flags):
                if c_flags[idx] == "-fopenmp":
                    c_flags.insert(idx, "-Xpreprocessor")
                    idx += 1
                idx += 1

        setup_code = create_c_setup(sharedlib_modname, wrapper_filename,
                dep_mods, compiler, includes, libs + extra_libs, libdirs + extra_libdirs, c_flags)
        setup_filename = "setup_{}.py".format(module_name)

        with open(setup_filename, 'w') as f:
            f.writelines(setup_code)

        setup_filename = os.path.join(pyccel_dirpath, setup_filename)
        cmd = [sys.executable, setup_filename, "build"]
        if sys.platform == 'win32' and compiler in ('gfortran','gcc','ifort','icc'):
            cmd += ['--compiler=mingw32']

        if verbose:
            print(' '.join(cmd))
        locks = [FileLock(d+'.lock') for d in dep_mods]
        for l in locks:
            l.acquire()
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            out, err = p.communicate()
        finally:
            for l in locks:
                l.release()
        if verbose:
            print(out)
        if p.returncode != 0:
            err_msg = "Failed to build module"
            err_msg += "\n" + err
            raise RuntimeError(err_msg)
        if err:
            warnings.warn(UserWarning(err))

        sharedlib_folder += 'build/lib*/'

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
