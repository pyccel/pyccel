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

# get path to pyccel/stdlib/lib_name
stdlib_path = os.path.dirname(stdlib_folder.__file__)

__all__ = ['copy_internal_library','recompile_object']

#==============================================================================
language_extension = {'fortran':'f90', 'c':'c', 'python':'py'}

#==============================================================================
def copy_internal_library(lib_folder, pyccel_dirpath):
    """
    Copy an internal library from its specified stdlib folder to the pyccel
    directory. The copy is only done if the files are not already present or
    if the files have changed since they were last copied

    Parameters
    ----------
    lib_folder     : str
                     The name of the folder to be copied, relative to the stdlib folder
    pyccel_dirpath : str
                     The location that the folder should be copied to

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
    compile_obj.acquire_lock()
    if os.path.exists(compile_obj.target):
        # Check if source file has changed since last compile
        o_file_age   = os.path.getmtime(compile_obj.target)
        src_file_age = os.path.getmtime(compile_obj.source)
        outdated     = o_file_age < src_file_age
    else:
        outdated = True
    if outdated:
        compiler.compile_module(compile_obj=compile_obj,
                output_folder=compile_obj.source_folder,
                verbose=verbose)
    compile_obj.release_lock()
