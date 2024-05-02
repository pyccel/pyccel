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
from filelock import FileLock
import pyccel.stdlib as stdlib_folder

from .compiling.basic     import CompileObj

# get path to pyccel/stdlib/lib_name
stdlib_path = os.path.dirname(stdlib_folder.__file__)

__all__ = ['copy_internal_library','recompile_object']

#==============================================================================
language_extension = {'fortran':'f90', 'c':'c', 'python':'py'}

#==============================================================================
# map internal libraries to their folders inside pyccel/stdlib and their compile objects
# The compile object folder will be in the pyccel dirpath
internal_libs = {
    "ndarrays"     : ("ndarrays", CompileObj("ndarrays.c",folder="ndarrays")),
    "pyc_math_f90" : ("math", CompileObj("pyc_math_f90.f90",folder="math")),
    "pyc_math_c"   : ("math", CompileObj("pyc_math_c.c",folder="math")),
    "cwrapper"     : ("cwrapper", CompileObj("cwrapper.c",folder="cwrapper", accelerators=('python',))),
}
internal_libs["cwrapper_ndarrays"] = ("cwrapper", CompileObj("cwrapper_ndarrays.c",folder="cwrapper",
                                                             accelerators = ('python',),
                                                             dependencies = (internal_libs["ndarrays"][1],)))

#==============================================================================
def copy_internal_library(lib_folder, pyccel_dirpath, extra_files = None):
    """
    Copy an internal library from its specified stdlib folder to the pyccel
    directory. The copy is only done if the files are not already present or
    if the files have changed since they were last copied. Extra files can be
    added to the folder if and when the copy occurs (e.g. for specifying
    the numpy version compatibility)

    Parameters
    ----------
    lib_folder     : str
                     The name of the folder to be copied, relative to the stdlib folder
    pyccel_dirpath : str
                     The location that the folder should be copied to
    extra_files    : dict
                     A dictionary whose keys are the names of any files to be created
                     in the folder and whose values are the contents of the file

    Results
    -------
    lib_dest_path  : str
                     The location that the files were copied to
    """
    # get lib path (stdlib_path/lib_name)
    lib_path = os.path.join(stdlib_path, lib_folder)
    # remove library folder to avoid missing files and copy
    # new one from pyccel stdlib
    lib_dest_path = os.path.join(pyccel_dirpath, lib_folder)
    with FileLock(lib_dest_path + '.lock'):
        to_copy = False
        if not os.path.exists(lib_dest_path):
            to_copy = True
        else:
            src_files = os.listdir(lib_path)
            dst_files = os.listdir(lib_dest_path)
            outdated = any(s not in dst_files for s in src_files)
            if not outdated:
                outdated = any(os.path.getmtime(os.path.join(lib_path, s)) > os.path.getmtime(os.path.join(lib_dest_path,s)) for s in src_files)
            if outdated:
                shutil.rmtree(lib_dest_path)
                to_copy = True
        if to_copy:
            shutil.copytree(lib_path, lib_dest_path)
            if extra_files:
                for filename, contents in extra_files.items():
                    with open(os.path.join(lib_dest_path, filename), 'w') as f:
                        f.writelines(contents)
    return lib_dest_path

#==============================================================================
def recompile_object(compile_obj,
                   compiler,
                   verbose = False):
    """
    Compile the provided file.
    If the file has already been compiled then it will only be recompiled
    if the source has been modified

    Parameters
    ----------
    compile_obj : CompileObj
                  The object to compile
    compiler    : str
                  The compiler used
    verbose     : bool
                  Indicates whethere additional information should be printed
    """

    # compile library source files
    compile_obj.acquire_simple_lock()
    if os.path.exists(compile_obj.module_target):
        # Check if source file has changed since last compile
        o_file_age   = os.path.getmtime(compile_obj.module_target)
        src_file_age = os.path.getmtime(compile_obj.source)
        outdated     = o_file_age < src_file_age
    else:
        outdated = True
    compile_obj.release_simple_lock()
    if outdated:
        compiler.compile_module(compile_obj=compile_obj,
                output_folder=compile_obj.source_folder,
                verbose=verbose)
