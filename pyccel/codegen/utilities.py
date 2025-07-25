# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#

"""
This file contains some useful functions to compile the generated fortran code
"""

import os
from pathlib import Path
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
stdlib_path = Path(stdlib_folder.__file__).parent

# get path to pyccel/extensions/lib_name
ext_path = Path(ext_folder.__file__).parent

# get path to pyccel/
pyccel_root = Path(__file__).parent.parent

errors = Errors()

__all__ = ['copy_internal_library','recompile_object']

#==============================================================================
language_extension = {'fortran':'f90', 'c':'c', 'python':'py'}

#==============================================================================
# map internal libraries to their folders inside pyccel/stdlib and their compile objects
# The compile object folder will be in the pyccel dirpath
internal_libs = {
    "pyc_math_f90"     : (stdlib_path / "math", "math", CompileObj("pyc_math_f90.f90",dirpath="math", libs = ('m',))),
    "pyc_math_c"       : (stdlib_path / "math", "math", CompileObj("pyc_math_c.c",dirpath="math")),
    "pyc_tools_f90"    : (stdlib_path / "tools", "tools", CompileObj("pyc_tools_f90.f90",dirpath="tools")),
    "cwrapper"         : (stdlib_path / "cwrapper", "cwrapper", CompileObj("cwrapper.c",dirpath="cwrapper", extra_compilation_tools=('python',))),
    "CSpan_extensions" : (stdlib_path / "STC_Extensions", "STC_Extensions", CompileObj("CSpan_extensions.h", dirpath="STC_Extensions", has_target_file = False)),
    "stc" : (ext_path / "STC/include/stc", "STC/include/stc", CompileObj("stc", dirpath="STC/include", has_target_file = False)),
    "gFTL" : (ext_path / "gFTL/install/GFTL-1.13/include/v2", "gFTL", CompileObj("gFTL", dirpath=".", has_target_file = False)),
}
internal_libs["STC_Extensions/Set_extensions"] = (stdlib_path / "STC_Extensions", "STC_Extensions", CompileObj("Set_Extensions.h",
                                                      dirpath="STC_Extensions",
                                                      has_target_file = False,
                                                      dependencies = (internal_libs['stc'][2],)))
internal_libs["STC_Extensions/Dict_extensions"] = (stdlib_path / "STC_Extensions", "STC_Extensions", CompileObj("Dict_Extensions.h",
                                                     dirpath="STC_Extensions",
                                                     has_target_file=False,
                                                     dependencies=(internal_libs["stc"][2],)))
internal_libs["STC_Extensions/List_extensions"] = (stdlib_path / "STC_Extensions", "STC_Extensions", CompileObj("List_Extensions.h",
                                                      dirpath="STC_Extensions",
                                                      has_target_file = False,
                                                      dependencies = (internal_libs['stc'][2],)))
internal_libs["STC_Extensions/Common_extensions"] = (stdlib_path / "STC_Extensions", "STC_Extensions", CompileObj("Common_Extensions.h",
                                                      dirpath="STC_Extensions",
                                                      has_target_file = False,
                                                      dependencies = (internal_libs['stc'][2],)))
internal_libs["gFTL_functions/Set_extensions"] = (stdlib_path / "gFTL_functions", "gFTL_functions", CompileObj("Set_Extensions.inc",
                                                                     dirpath="gFTL_functions",
                                                                     has_target_file = False,
                                                                     dependencies = (internal_libs['gFTL'][2],)))
internal_libs["gFTL_functions/Vector_extensions"] = (stdlib_path / "gFTL_functions", "gFTL_functions", CompileObj("Vector_Extensions.inc",
                                                                     dirpath="gFTL_functions",
                                                                     has_target_file = False,
                                                                     dependencies = (internal_libs['gFTL'][2],)))
internal_libs["gFTL_functions/Map_extensions"] = (stdlib_path / "gFTL_functions", "gFTL_functions", CompileObj("Map_Extensions.inc",
                                                                     dirpath="gFTL_functions",
                                                                     has_target_file = False,
                                                                     dependencies = (internal_libs['gFTL'][2],)))
internal_libs["stc/cstr"] = (ext_path / "STC/src", "STC/src", CompileObj("cstr_core.c", dirpath="STC/include", dependencies = (internal_libs['stc'][2],)))
internal_libs["stc/cspan"] = (ext_path / "STC/src", "STC/src", CompileObj("cspan.c", dirpath="STC/include", dependencies = (internal_libs['stc'][2],)))
internal_libs["stc/algorithm"] = (ext_path / "STC/include/stc/", "STC/include/stc/", CompileObj("algorithm.h",
                                    dirpath="STC/include",
                                    has_target_file = False,
                                    dependencies = (internal_libs['stc'][2],)))


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
def copy_internal_library(dst_folder, lib_path, pyccel_dirpath, *, extra_files = None):
    """
    Copy an internal library to the specified Pyccel directory.

    Copy an internal library from its specified stdlib folder to the Pyccel
    directory. The copy is only done if the files are not already present or
    if the files have changed since they were last copied. Extra files can be
    added to the folder if and when the copy occurs (e.g. for specifying
    the NumPy version compatibility).

    Parameters
    ----------
    dst_folder : str
        The name of the folder to be copied to, relative to the __pyccel__ folder.

    lib_path : str
        The absolute path to the folder to be copied.

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

    # remove library folder to avoid missing files and copy
    # new one from pyccel stdlib
    lib_dest_path = os.path.join(pyccel_dirpath, dst_folder)
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
                               compiler, include, libs, libdir, dependencies,
                               extra_compilation_tools, language, verbose, convert_only):
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
    include : iterable of strs
        Include directories paths.
    libs : iterable of strs
        Required libraries.
    libdir : iterable of strs
        Paths to directories containing the required libraries.
    dependencies : iterable of CompileObjs
        Objects which must also be compiled in order to compile this module/program.
    extra_compilation_tools : iterable of str
        Tools used which require additional compilation flags/include dirs/libs/etc.
    language : str
        The language in which code is being printed.
    verbose : int
        Indicates the level of verbosity.
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
        printer = printer_registry[language](filename, verbose = verbose)
        code = printer.doprint(mod)
        if not os.path.exists(folder):
            os.mkdir(folder)
        with FileLock(f'{folder}.lock'):
            with open(filename, 'w', encoding="utf-8") as f:
                f.write(code)

        new_dependencies.append(CompileObj(os.path.basename(filename), dirpath=folder,
                            include=include,
                            libs=libs, libdir=libdir,
                            dependencies=(*dependencies, internal_libs['gFTL'][2]),
                            extra_compilation_tools=extra_compilation_tools))
        manage_dependencies({'gFTL':None}, compiler, pyccel_dirpath, new_dependencies[-1],
                language, verbose, convert_only)

    return new_dependencies

#==============================================================================
def recompile_object(compile_obj,
                   compiler,
                   language,
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

    language : str
        The language in which code is being printed.

    verbose : int
        Indicates the level of verbosity.
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
                output_dirpath=compile_obj.source_dirpath,
                language=language,
                verbose=verbose)

#==============================================================================
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
    verbose : int
        Indicates the level of verbosity.
    convert_only : bool, default=False
        Indicates if the compilation step is required or not.
    """
    # Iterate over the internal_libs list and determine if the printer
    # requires an internal lib to be included.
    for lib_name, (src_folder, dst_folder, stdlib) in internal_libs.items():
        if lib_name in pyccel_imports:
            extra_files = {'numpy_version.h' : get_numpy_max_acceptable_version_file()} \
                        if lib_name == 'cwrapper' else None

            lib_dest_path = copy_internal_library(dst_folder, src_folder, pyccel_dirpath, extra_files = extra_files)
            if lib_name == 'stc':
                lib_dest_path = os.path.dirname(lib_dest_path)
            # Pylint thinks stdlib is a str
            if stdlib.dependencies: # pylint: disable=E1101
                manage_dependencies({os.path.splitext(os.path.basename(d.source))[0]: None for d in stdlib.dependencies}, # pylint: disable=E1101
                        compiler, pyccel_dirpath, stdlib, language, verbose, convert_only)

            # stop after copying lib to __pyccel__ directory for
            # convert only
            if convert_only:
                continue
            stdlib.reset_dirpath(lib_dest_path) # pylint: disable=E1101

            # get the include folder path and library files
            recompile_object(stdlib,
                             compiler = compiler,
                             language = language,
                             verbose  = verbose)

            mod_obj.add_dependencies(stdlib)

    # Iterate over the external_libs list and determine if the printer
    # requires an external lib to be included.
    for key, import_node in pyccel_imports.items():
        deps = generate_extension_modules(key, import_node, pyccel_dirpath,
                                          compiler     = compiler,
                                          include     = mod_obj.include,
                                          libs         = mod_obj.libs,
                                          libdir      = mod_obj.libdir,
                                          dependencies = mod_obj.dependencies,
                                          extra_compilation_tools = mod_obj.extra_compilation_tools,
                                          language = language,
                                          verbose = verbose,
                                          convert_only = convert_only)
        for d in deps:
            recompile_object(d,
                             compiler = compiler,
                             language = language,
                             verbose  = verbose)
            mod_obj.add_dependencies(d)

#==============================================================================
def get_module_and_compile_dependencies(parser, compile_libs = None, deps = None):
    """
    Get the module (.o files) and compilation dependencies.

    Determine all additional .o files, include folders and libraries required
    to generate the shared library or executable.

    Parameters
    ----------
    parser : Parser
        The parser whose dependencies should be appended.
    compile_libs : list[str], optional
        The libraries (-lX) that should be used for the compilation.
        This argument is used internally but should not be provided
        from an external call to this function.
    deps : dict[str, CompileObj], optional
        A dictionary describing the modules on which this code depends.
        The key is the name of the file containing the module. The value
        is the CompileObj describing the .o file.
        This argument is used internally but should not be provided
        from an external call to this function.

    Returns
    -------
    compile_libs : list[str], optional
        The libraries (-lX) that should be used for the compilation.
    deps : dict[str, CompileObj], optional
        A dictionary describing the modules on which this code depends.
        The key is the name of the file containing the module. The value
        is the CompileObj describing the .o file.
    """
    dep_fname = Path(parser.filename)
    assert compile_libs is None or dep_fname.suffix in ('.pyi', '.pyh') or pyccel_root in dep_fname.parents
    mod_folder = dep_fname.parent
    mod_base = dep_fname.name

    if compile_libs is None:
        assert deps is None
        compile_libs = []
        deps = {}
    else:
        # Stop conditions
        if parser.metavars.get('module_name', None) == 'omp_lib':
            return compile_libs, deps

        if parser.compile_obj:
            deps[dep_fname] = parser.compile_obj
        elif dep_fname not in deps:
            dep_compile_libs = [l for l in parser.metavars.get('libraries', '').split(',') if l]
            if not parser.metavars.get('ignore_at_import', False):
                deps[dep_fname] = CompileObj(mod_base,
                                    dirpath         = mod_folder,
                                    libs            = dep_compile_libs,
                                    has_target_file = not parser.metavars.get('no_target', False))
            else:
                compile_libs.extend(dep_compile_libs)

    # Proceed recursively
    for son in parser.sons:
        get_module_and_compile_dependencies(son, compile_libs, deps)

    return compile_libs, deps
