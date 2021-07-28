# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
This file contains some useful functions to compile the generated fortran code
"""

import os
import shutil
import subprocess
import sys
import sysconfig
import warnings
from filelock import FileLock
import pyccel.stdlib as stdlib_folder
from .compiling.basic     import CompileObj

# get path to pyccel/stdlib/lib_name
stdlib_path = os.path.dirname(stdlib_folder.__file__)

__all__ = ['construct_flags', 'compile_files', 'get_gfortran_library_dir']

#==============================================================================
# TODO use constructor and a dict to map flags w.r.t the compiler
_avail_compilers = ['gfortran', 'mpif90', 'pgfortran', 'ifort', 'gcc', 'icc']

language_extension = {'fortran':'f90', 'c':'c', 'python':'py'}

#==============================================================================
# TODO add opt flags, etc... look at f2py interface in numpy
def construct_flags(compiler,
                    fflags=(),
                    debug=False,
                    accelerator=None,
                    includes=()):
    """
    Constructs compiling flags for a given compiler.

    fflags: list
        Fortran compiler flags. Default is `-O3`

    compiler: str
        used compiler for the target language.

    accelerator: str
        name of the selected accelerator.
        One among ('openmp', 'openacc')

    debug: bool
        add some useful prints that may help for debugging.

    includes: list
        list of include directories paths
    """

    if not(compiler in _avail_compilers):
        raise ValueError("Only {0} are available.".format(_avail_compilers))

    if not fflags:
        flags = ['-O3']
    else:
        flags = list(fflags)

    mac_target = sysconfig.get_config_var('MACOSX_DEPLOYMENT_TARGET')
    if mac_target:
        os.environ['MACOSX_DEPLOYMENT_TARGET'] = mac_target

    if compiler == "gfortran":
        if debug:
            flags.append("-fcheck=bounds")

    if compiler == "icc":
        flags.append("-std=c99")

    if compiler == "mpif90":
        if debug:
            flags.append("-fcheck=bounds")
        if sys.platform == "win32":
            mpiinc = os.environ["MSMPI_INC"].rstrip('\\')
            mpilib = os.environ["MSMPI_LIB64"].rstrip('\\')
            flags.extend(['-D', 'USE_MPI_MODULE', '-I', mpiinc, '-L', mpilib])

    if accelerator is not None:
        if accelerator == "openmp":
            if sys.platform == "darwin" and compiler == "gcc":
                flags.append("-Xpreprocessor")

            flags.append("-fopenmp")
            if compiler == 'ifort':
                flags.append('-nostandard-realloc-lhs')

        elif accelerator == "openacc":
            flags.extend(["-ta=multicore", "-Minfo=accel"])
        else:
            raise ValueError("Only openmp and openacc are available")

    # Construct flags
    flags.extend(f for i in includes for f in ('-I', i))

    return flags

#==============================================================================
def compile_files(filename, compiler, flags,
                    binary=None,
                    verbose=False,
                    modules=(),
                    is_module=False,
                    libs=(),
                    libdirs=(),
                    language="fortran",
                    output=''):
    """
    Compiles the generated file.

    verbose: bool
        talk more
    """
    flags=list(flags)

    if binary is None:
        if not is_module:
            binary = os.path.splitext(os.path.basename(filename))[0]
        else:
            f = os.path.join(output, os.path.splitext(os.path.basename(filename))[0])
            binary = '{}.o'.format(f)
#            binary = "{folder}{binary}.o".format(folder=output,
#                                binary=os.path.splitext(os.path.basename(filename))[0])

    o_code = '-o'
    j_code = []
    if is_module:
        flags.append('-c')
        if (len(output)>0) and language == "fortran":
            if compiler == "ifort":
                j_code = ['-module', output]
            else:
                j_code = ['-J', output]

    m_code = ['{}.o'.format(m) for m in modules]
    if is_module:
        libs_flags = []
    else:
        flags.extend(f for i in libdirs for f in ('-L', i))
        libs_flags = ['-l{}'.format(i) for i in libs]

    if sys.platform == "win32" and compiler == "mpif90":
        compiler = "gfortran"
        m_code.append(os.path.join(os.environ["MSMPI_LIB64"], 'libmsmpi.a'))

    cmd = [compiler, *flags, *m_code, filename, o_code, binary, *libs_flags, *j_code]

    if verbose:
        print(' '.join(cmd))

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    out, err = p.communicate()

    if verbose and out:
        print(out)
    if p.returncode != 0:
        err_msg = "Failed to build module"
        err_msg += "\n" + err
        raise RuntimeError(err_msg)
    if err:
        warnings.warn(UserWarning(err))

    # TODO shall we uncomment this?
#    # write and save a log file in .pyccel/'filename'.log
#    # ...
#    def mkdir_p(dir):
#        # type: (unicode) -> None
#        if os.path.isdir(dir):
#            return
#        os.makedirs(dir)
#
#    if True:
#        tmp_dir = '.pyccel'
#        mkdir_p(tmp_dir)
#        logfile = '{0}.log'.format(binary)
#        logfile = os.path.join(tmp_dir, logfile)
#        f = open(logfile, 'w')
#        f.write(output)
#        f.close()

    return output, cmd

def get_gfortran_library_dir():
    """Provide the location of the gfortran libraries for linking
    """
    if sys.platform == "win32":
        file_name_list = ['gfortran.lib', 'libgfortran.a']
    else:
        file_name_list = ['libgfortran.a']

    for file_name in file_name_list:
        file_location = subprocess.check_output([shutil.which('gfortran'), '-print-file-name='+file_name],
                universal_newlines = True)
        lib_dir = os.path.abspath(os.path.dirname(file_location))
        if lib_dir != os.getcwd():
            if lib_dir not in sys.path:
                # Add to system path
                sys.path.insert(0, lib_dir)
    return lib_dir

def copy_internal_library(lib_folder, pyccel_dirpath):
    # get lib path (stdlib_path/lib_name)
    lib_path = os.path.join(stdlib_path, lib_folder)
    # remove library folder to avoid missing files and copy
    # new one from pyccel stdlib
    lib_dest_path = os.path.join(pyccel_dirpath, lib_folder)
    with FileLock(lib_dest_path + '.lock'):
        if not os.path.exists(lib_dest_path):
            shutil.copytree(lib_path, lib_dest_path)
        else:
            src_files = os.listdir(lib_path)
            dst_files = os.listdir(lib_dest_path)
            outdated = any(s not in dst_files for s in src_files)
            if not outdated:
                outdated = any(os.path.getmtime(os.path.join(lib_path, s)) > os.path.getmtime(os.path.join(lib_dest_path,s)) for s in src_files)
            if outdated:
                shutil.rmtree(lib_dest_path)
                shutil.copytree(lib_path, lib_dest_path)
    return lib_dest_path

def compile_folder(folder,
                   language,
                   compiler,
                   debug = False,
                   verbose = False):
    """
    Compile all files matching the language extension in a folder

    Parameters
    ----------
    folder   : str
               The folder to compile
    language : str
               The language we are translating to
    compiler : str
               The compiler used
    includes : iterable
               Any folders which should be added to the default includes
    libs     : iterable
               Any libraries which are needed to compile
    libdirs  : iterable
               Any folders which should be added to the default libdirs
    debug    : bool
               Indicates whether we should compile in debug mode
    verbose  : bool
               Indicates whethere additional information should be printed
    """

    # get library source files
    ext = '.'+language_extension[language]
    source_files = [os.path.join(folder, e) for e in os.listdir(folder)
                                                if e.endswith(ext)]
    compile_objs = [CompileObj(s,folder) for s in source_files]

    # compile library source files
    for f in compile_objs:
        f.acquire_lock()
        if os.path.exists(f.target):
            o_file_age   = os.path.getmtime(f.target)
            src_file_age = os.path.getmtime(f.source)
            outdated     = o_file_age < src_file_age
        else:
            outdated = True
        if outdated:
            compiler.compile_file(compile_obj=f,
                    output_folder=folder,
                    verbose=verbose)
        f.release_lock()
    return compile_objs
