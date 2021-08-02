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

__all__ = ['copy_internal_library','compile_folder']

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
def compile_folder(folder,
                   language,
                   compiler,
                   debug = False,
                   verbose = False):
    """
    Compile all files matching the language extension in a folder.
    If the file has already been compiled then it will only be recompiled
    if the source has been modified

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
            # Check if source file has changed since last compile
            o_file_age   = os.path.getmtime(f.target)
            src_file_age = os.path.getmtime(f.source)
            outdated     = o_file_age < src_file_age
        else:
            outdated = True
        if outdated:
            compiler.compile_module(compile_obj=f,
                    output_folder=folder,
                    verbose=verbose)
        f.release_lock()
    return compile_objs
