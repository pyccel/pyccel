# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#

"""
This file contains some useful functions to compile the generated fortran code
"""

import os
import shutil
from filelock import FileLock
import pyccel.stdlib as stdlib_folder
import pyccel.extensions as ext_folder

from .codegen              import printer_registry
from .compiling.basic      import CompileObj
from .compiling.file_locks import FileLockSet

# get path to pyccel/stdlib/lib_name
stdlib_path = os.path.dirname(stdlib_folder.__file__)

# get path to pyccel/extensions/lib_name
ext_path = os.path.dirname(ext_folder.__file__)

__all__ = ['copy_internal_library','recompile_object']

#==============================================================================
language_extension = {'fortran':'f90', 'c':'c', 'python':'py'}

#==============================================================================
# map external libraries inside pyccel/extensions with their path
external_libs = {"stc" : "STC/include/stc", "gFTL" : "gFTL/install/GFTL-1.13/include/v2"}

#==============================================================================
# map internal libraries to their folders inside pyccel/stdlib and their compile objects
# The compile object folder will be in the pyccel dirpath
internal_libs = {
    "ndarrays"        : ("ndarrays", CompileObj("ndarrays.c",folder="ndarrays")),
    "pyc_math_f90"    : ("math", CompileObj("pyc_math_f90.f90",folder="math")),
    "pyc_math_c"      : ("math", CompileObj("pyc_math_c.c",folder="math")),
    "pyc_tools_f90"   : ("tools", CompileObj("pyc_tools_f90.f90",folder="tools")),
    "cwrapper"        : ("cwrapper", CompileObj("cwrapper.c",folder="cwrapper", accelerators=('python',))),
    "numpy_f90"       : ("numpy", CompileObj("numpy_f90.f90",folder="numpy")),
    "numpy_c"         : ("numpy", CompileObj("numpy_c.c",folder="numpy")),
    "Set_extensions"  : ("STC_Extensions", CompileObj("Set_Extensions.h", folder="STC_Extensions", has_target_file = False)),
    "List_extensions" : ("STC_Extensions", CompileObj("List_Extensions.h", folder="STC_Extensions", has_target_file = False)),
}
internal_libs["cwrapper_ndarrays"] = ("cwrapper_ndarrays", CompileObj("cwrapper_ndarrays.c",folder="cwrapper_ndarrays",
                                                             accelerators = ('python',),
                                                             dependencies = (internal_libs["ndarrays"][1],
                                                                             internal_libs["cwrapper"][1])))

#==============================================================================

def not_a_copy(src_folder, dst_folder, filename):
    """
    Check if the file is different between the source and destination folders.

    Check if the file filename present in src_folder is different (not a copy)
    from the file filename present in dst_folder. This is done by checking if
    the source file has been modified more recently than the destination file.
    This would imply that it has been modified since the last copy.

    Parameters
    ----------
    src_folder : str
        The folder where the file was defined.

    dst_folder : str
        The folder where the file is being used.

    filename : str
        The name of the file.

    Returns
    -------
    bool
        False if the file in the destination folder is a copy of the file in the
        source folder, True otherwise.
    """
    abs_src_file = os.path.join(src_folder, filename)
    abs_dst_file = os.path.join(dst_folder, filename)
    src_mod_time = os.path.getmtime(abs_src_file)
    dst_mod_time = os.path.getmtime(abs_dst_file)
    return src_mod_time > dst_mod_time

#==============================================================================
def copy_internal_library(lib_folder, pyccel_dirpath, extra_files = None):
    """
    Copy an internal library to the specified Pyccel directory.

    Copy an internal library from its specified stdlib folder to the Pyccel
    directory. The copy is only done if the files are not already present or
    if the files have changed since they were last copied. Extra files can be
    added to the folder if and when the copy occurs (e.g. for specifying
    the NumPy version compatibility).

    Parameters
    ----------
    lib_folder : str
        The name of the folder to be copied, relative to the stdlib folder.

    pyccel_dirpath : str
        The location that the folder should be copied to.

    extra_files : dict
        A dictionary whose keys are the names of any files to be created
        in the folder and whose values are the contents of the file.

    Returns
    -------
    str
        The location that the files were copied to.
    """
    # get lib path (stdlib_path/lib_name or ext_path/lib_name)
    if lib_folder in external_libs:
        lib_path = os.path.join(ext_path, external_libs[lib_folder])
    else:
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
            src_files = [os.path.relpath(os.path.join(root, f), lib_path) \
                    for root, dirs, files in os.walk(lib_path) for f in files]
            dst_files = [os.path.relpath(os.path.join(root, f), lib_dest_path) \
                    for root, dirs, files in os.walk(lib_dest_path) \
                    for f in files if not f.endswith('.lock')]
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
            locks = FileLockSet()
            for s in src_files:
                base, ext = os.path.splitext(s)
                if ext != '.h':
                    locks.append(FileLock(os.path.join(lib_dest_path, base+'.o.lock')))
            # Acquire locks to avoid compilation problems
            with locks:
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
    return lib_dest_path

#==============================================================================
def generate_extension_modules(import_key, import_node, pyccel_dirpath,
                               includes, libs, libdirs, dependencies,
                               accelerators, language):
    """
    Generate any new modules that describe extensions.

    Generate any new modules that describe extensions. This is the case for lists/
    sets/dicts/etc handled by gFTL.

    Parameters
    ----------
    import_key : str
        The name by which the extension is identified in the import.
    import_node : Import
        The import used in the code generator (this object contains the module to
        be printed).
    pyccel_dirpath : str
        The folder where files are being saved.
    includes : iterable of strs
        Include directories paths.
    libs : iterable of strs
        Required libraries.
    libdirs : iterable of strs
        Paths to directories containing the required libraries.
    dependencies : iterable of CompileObjs
        Objects which must also be compiled in order to compile this module/program.
    accelerators : iterable of str
        Tool used to accelerate the code (e.g. openmp openacc).
    language : str
        The language in which code is being printed.

    Returns
    -------
    list[CompileObj]
        A list of any new compilation dependencies which are required to compile
        the translated file.
    """
    new_dependencies = []
    lib_name = import_key.split('/', 1)[0]
    if lib_name == 'gFTL_extensions':
        lib_name = 'gFTL'
        mod = import_node.source_module
        printer = printer_registry[language]
        filename = os.path.join(pyccel_dirpath, import_key)+'.F90'
        folder = os.path.dirname(filename)
        code = printer(filename).doprint(mod)
        if not os.path.exists(folder):
            os.mkdir(folder)
        with FileLock(f'{folder}.lock'):
            with open(filename, 'w', encoding="utf-8") as f:
                f.write(code)

        new_dependencies.append(CompileObj(os.path.basename(filename), folder=folder,
                            includes=(os.path.join(pyccel_dirpath, 'gFTL'), *includes),
                            libs=libs, libdirs=libdirs, dependencies=dependencies,
                            accelerators=accelerators))

    if lib_name in external_libs:
        copy_internal_library(lib_name, pyccel_dirpath)

    return new_dependencies

#==============================================================================
def recompile_object(compile_obj,
                   compiler,
                   verbose = False):
    """
    Compile the provided file if necessary.

    Check if the file has already been compiled, if it hasn't or if the source has
    been modified then compile the file.

    Parameters
    ----------
    compile_obj : CompileObj
        The object to compile.

    compiler : str
        The compiler used.

    verbose : bool
        Indicates whether additional information should be printed.
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
