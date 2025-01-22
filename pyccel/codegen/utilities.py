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
from pyccel.errors.errors import Errors

from pyccel.ast.numpy_wrapper                    import get_numpy_max_acceptable_version_file

from .codegen              import printer_registry
from .compiling.basic      import CompileObj
from .compiling.file_locks import FileLockSet

# get path to pyccel/stdlib/lib_name
stdlib_path = os.path.dirname(stdlib_folder.__file__)

# get path to pyccel/extensions/lib_name
ext_path = os.path.dirname(ext_folder.__file__)

errors = Errors()

__all__ = ['copy_internal_library','recompile_object']

#==============================================================================
language_extension = {'fortran':'f90', 'c':'c', 'python':'py'}

#==============================================================================
# map external libraries inside pyccel/extensions with their path
external_libs = {"stc" : ("STC/include/stc", CompileObj("stc", folder="stc", has_target_file = False)),
                 "gFTL" : ("gFTL/install/GFTL-1.13/include/v2", CompileObj("gFTL", folder="gFTL", has_target_file = False)),
}
#==============================================================================
# map internal libraries to their folders inside pyccel/stdlib and their compile objects
# The compile object folder will be in the pyccel dirpath
internal_libs = {
    "pyc_math_f90"     : ("math", CompileObj("pyc_math_f90.f90",folder="math")),
    "pyc_math_c"       : ("math", CompileObj("pyc_math_c.c",folder="math")),
    "pyc_tools_f90"    : ("tools", CompileObj("pyc_tools_f90.f90",folder="tools")),
    "cwrapper"         : ("cwrapper", CompileObj("cwrapper.c",folder="cwrapper", accelerators=('python',))),
    "CSpan_extensions" : ("STC_Extensions", CompileObj("CSpan_extensions.h", folder="STC_Extensions", has_target_file = False)),
    "Set_extensions"   : ("STC_Extensions", CompileObj("Set_Extensions.h",
                                                      folder="STC_Extensions",
                                                      has_target_file = False,
                                                      dependencies = (external_libs['stc'][1],))),
    "List_extensions" : ("STC_Extensions", CompileObj("List_Extensions.h",
                                                      folder="STC_Extensions",
                                                      has_target_file = False,
                                                      dependencies = (external_libs['stc'][1],))),
    "Common_extensions" : ("STC_Extensions", CompileObj("Common_Extensions.h",
                                                      folder="STC_Extensions",
                                                      has_target_file = False,
                                                      dependencies = (external_libs['stc'][1],))),
    "gFTL_functions/Set_extensions"  : ("gFTL_functions", CompileObj("Set_Extensions.inc",
                                                                     folder="gFTL_functions",
                                                                     has_target_file = False,
                                                                     dependencies = (external_libs['gFTL'][1],))),
    "stc/cstr" : ("STC_Extensions", CompileObj("cstr.c", folder="STC_Extensions", dependencies = (external_libs['stc'][1],)))
}

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
        lib_path = os.path.join(ext_path, external_libs[lib_folder][0])
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

    if lib_folder in external_libs and lib_folder != 'gFTL':
        return pyccel_dirpath
    else:
        return lib_dest_path

#==============================================================================
def generate_extension_modules(import_key, import_node, pyccel_dirpath,
                               compiler, includes, libs, libdirs, dependencies,
                               accelerators, language, verbose, convert_only):
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
    compiler : Compiler
        A compiler that can be used to compile dependencies.
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
    verbose : bool
        Indicates whether additional information should be printed.
    convert_only : bool, default=False
        Indicates if the compilation step is required or not.

    Returns
    -------
    list[CompileObj]
        A list of any new compilation dependencies which are required to compile
        the translated file.
    """
    new_dependencies = []
    lib_name = str(import_key).split('/', 1)[0]
    if lib_name == 'gFTL_extensions':
        lib_name = 'gFTL'
        mod = import_node.source_module
        filename = os.path.join(pyccel_dirpath, import_key)+'.F90'
        folder = os.path.dirname(filename)
        printer = printer_registry[language](filename)
        code = printer.doprint(mod)
        if not os.path.exists(folder):
            os.mkdir(folder)
        with FileLock(f'{folder}.lock'):
            with open(filename, 'w', encoding="utf-8") as f:
                f.write(code)

        new_dependencies.append(CompileObj(os.path.basename(filename), folder=folder,
                            includes=includes,
                            libs=libs, libdirs=libdirs,
                            dependencies=(*dependencies, external_libs['gFTL'][1]),
                            accelerators=accelerators))
        manage_dependencies({'gFTL':None}, compiler, pyccel_dirpath, new_dependencies[-1],
                language, verbose, convert_only)

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

def manage_dependencies(pyccel_imports, compiler, pyccel_dirpath, mod_obj, language, verbose, convert_only = False):
    """
    Manage dependencies of the code to be compiled.

    Manage dependencies of the code to be compiled.

    Parameters
    ----------
    pyccel_imports : dict[str,Import]
        A dictionary describing imports created by Pyccel that may imply dependencies.
    compiler : Compiler
        A compiler that can be used to compile dependencies.
    pyccel_dirpath : str
        The path in which the Pyccel output is generated (__pyccel__).
    mod_obj : CompileObj
        The object that we are aiming to copile.
    language : str
        The language in which code is being printed.
    verbose : bool
        Indicates whether additional information should be printed.
    convert_only : bool, default=False
        Indicates if the compilation step is required or not.
    """
    # Copy any necessary external libraries
    for import_key in pyccel_imports:
        lib_name = str(import_key).split('/', 1)[0]
        if lib_name in external_libs:
            lib_dest_path = copy_internal_library(lib_name, pyccel_dirpath)
            external_libs[lib_name][1].reset_folder(lib_dest_path)

    # Iterate over the internal_libs list and determine if the printer
    # requires an internal lib to be included.
    for lib_name, (stdlib_folder, stdlib) in internal_libs.items():
        if lib_name in pyccel_imports:

            extra_files = {'numpy_version.h' : get_numpy_max_acceptable_version_file()} \
                        if lib_name == 'cwrapper' else None

            lib_dest_path = copy_internal_library(stdlib_folder, pyccel_dirpath, extra_files)

            # Pylint thinks stdlib is a str
            if stdlib.dependencies: # pylint: disable=E1101
                manage_dependencies({os.path.splitext(os.path.basename(d.source))[0]: None for d in stdlib.dependencies}, # pylint: disable=E1101
                        compiler, pyccel_dirpath, stdlib, language, verbose, convert_only)

            # stop after copying lib to __pyccel__ directory for
            # convert only
            if convert_only:
                continue
            stdlib.reset_folder(lib_dest_path) # pylint: disable=E1101

            # get the include folder path and library files
            recompile_object(stdlib,
                             compiler = compiler,
                             verbose  = verbose)

            mod_obj.add_dependencies(stdlib)

    # Iterate over the external_libs list and determine if the printer
    # requires an external lib to be included.
    for key, import_node in pyccel_imports.items():
        deps = generate_extension_modules(key, import_node, pyccel_dirpath,
                                          compiler     = compiler,
                                          includes     = mod_obj.includes,
                                          libs         = mod_obj.libs,
                                          libdirs      = mod_obj.libdirs,
                                          dependencies = mod_obj.dependencies,
                                          accelerators = mod_obj.accelerators,
                                          language = language,
                                          verbose = verbose,
                                          convert_only = convert_only)
        for d in deps:
            recompile_object(d,
                             compiler = compiler,
                             verbose  = verbose)
            mod_obj.add_dependencies(d)

