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
    "numpy_f90"    : ("numpy", CompileObj("numpy_f90.f90",folder="numpy")),
    "numpy_c"      : ("numpy", CompileObj("numpy_c.c",folder="numpy")),
}
internal_libs["cwrapper_ndarrays"] = ("cwrapper_ndarrays", CompileObj("cwrapper_ndarrays.c",folder="cwrapper_ndarrays",
                                                             accelerators = ('python',),
                                                             dependencies = (internal_libs["ndarrays"][1],
                                                                             internal_libs["cwrapper"][1])))

#==============================================================================

def not_a_copy(src_folder, dst_folder, filename):
    """ Check if the file filename present in src_folder
    is a copy of the file filename present in dst_folder
    or if the source file has been updated since the last
    copy
    """
    abs_src_file = os.path.join(src_folder, filename)
    abs_dst_file = os.path.join(dst_folder, filename)
    src_mod_time = os.path.getmtime(abs_src_file)
    dst_mod_time = os.path.getmtime(abs_dst_file)
    return src_mod_time > dst_mod_time

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
        # Check if folder exists
        if not os.path.exists(lib_dest_path):
            to_create = True
            to_update = False
        else:
            to_create = False
            # If folder exists check if it needs updating
            src_files = os.listdir(lib_path)
            dst_files = [f for f in os.listdir(lib_dest_path) if not f.endswith('.lock')]
            # Check if all files are present in destination
            to_update = any(s not in dst_files for s in src_files)

            # Check if original files have been modified
            if not to_update:
                to_update = any(not_a_copy(lib_path, lib_dest_path, s) for s in src_files)

        if to_create:
            # Copy all files from the source to the destination
            shutil.copytree(lib_path, lib_dest_path)
            # Create any requested extra files
            if extra_files:
                for filename, contents in extra_files.items():
                    with open(os.path.join(lib_dest_path, filename), 'w') as f:
                        f.writelines(contents)
        elif to_update:
            locks = []
            for s in src_files:
                base, ext = os.path.splitext(s)
                if ext != '.h':
                    locks.append(FileLock(os.path.join(lib_dest_path, base+'.o.lock')))
            # Acquire locks to avoid compilation problems
            for l in locks:
                l.acquire()
            # Remove all files in destination directory
            for d in dst_files:
                d_file = os.path.join(lib_dest_path, d)
                try:
                    os.remove(d_file)
                except FileNotFoundError:
                    # Don't call error in case of temporary compilation file that has disappeared
                    # since reading the folder
                    pass
            # Copy all files from the source to the destination
            for s in src_files:
                shutil.copyfile(os.path.join(lib_path, s),
                        os.path.join(lib_dest_path, s))
            # Create any requested extra files
            if extra_files:
                for filename, contents in extra_files.items():
                    extra_file = os.path.join(lib_dest_path, filename)
                    with open(extra_file, 'w', encoding="utf-8") as f:
                        f.writelines(contents)
            # Release the locks
            for l in locks:
                l.release()
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
    with compile_obj:
        if os.path.exists(compile_obj.module_target):
            # Check if source file has changed since last compile
            o_file_age   = os.path.getmtime(compile_obj.module_target)
            src_file_age = os.path.getmtime(compile_obj.source)
            outdated     = o_file_age < src_file_age
        else:
            outdated = True
    if outdated:
        compiler.compile_module(compile_obj=compile_obj,
                output_folder=compile_obj.source_folder,
                verbose=verbose)
