# coding: utf-8
import os
import shutil

from pyccel.parser.errors     import Errors
from pyccel.codegen.codegen   import Parser
from pyccel.codegen.codegen   import Codegen
from pyccel.codegen.utilities import construct_flags
from pyccel.codegen.utilities import compile_fortran
from pyccel.codegen.f2py      import create_shared_library

__all__ = ['execute_pyccel']

#==============================================================================
# NOTE:
# [..]_dirname is the name of a directory
# [..]_dirpath is the full (absolute) path of a directory

# TODO: prune options
# TODO: change name of variable 'module_name', as it could be a program
# TODO [YG, 04.02.2020]: check if we should catch BaseException instead of Exception
def execute_pyccel(fname, *,
                   syntax_only   = False,
                   semantic_only = False,
                   convert_only  = False,
                   recursive     = False,
                   verbose       = False,
                   folder        = None,
                   compiler      = None,
                   mpi_compiler  = None,
                   fflags        = None,
                   includes      = (),
                   libdirs       = (),
                   modules       = (),
                   libs          = (),
                   debug         = False,
                   extra_args    = '',
                   accelerator   = None,
                   output_name   = None):

    # TODO [YG, 03.02.2020]: test validity of function arguments

    # Copy list arguments to local lists to avoid unexpected behavior
    includes = [*includes]
    libdirs  = [*libdirs]
    modules  = [*modules]
    libs     = [*libs]

    # Store current directory
    base_dirpath = os.getcwd()

    # Unified way to handle errors: print formatted error message, then move
    # to original working directory. Caller should then raise exception.
    def handle_error(stage):
        print('\nERROR at {} stage'.format(stage))
        os.chdir(base_dirpath)

    # Identify absolute path, directory, and filename
    pymod_filepath = os.path.abspath(fname)
    pymod_dirpath, pymod_filename = os.path.split(pymod_filepath)

    # Extract module name
    module_name = os.path.splitext(pymod_filename)[0]

    # Define working directory 'folder'
    if folder is None or folder == "":
        folder = pymod_dirpath
    else:
        folder = os.path.abspath(folder)

    # Define directory name and path for pyccel & f2py build
    pyccel_dirname = '__pyccel__'
    pyccel_dirpath = os.path.join(folder, pyccel_dirname)

    # Create new directories if not existing
    os.makedirs(folder, exist_ok=True)
    os.makedirs(pyccel_dirpath, exist_ok=True)

    # Change working directory to 'folder'
    os.chdir(folder)

    # Choose Fortran compiler
    if compiler is None:
        compiler = 'gfortran'

    f90exec = mpi_compiler if mpi_compiler else compiler

    # ...
    # Construct flags for the Fortran compiler
    if fflags is None:
        fflags = construct_flags(f90exec,
                                 fflags=None,
                                 debug=debug,
                                 accelerator=accelerator,
                                 includes=(),
                                 libdirs=())

    # Build position-independent code, suited for use in shared library
    fflags = ' {} -fPIC '.format(fflags)
    # ...

    # Parse Python file
    try:
        parser = Parser(pymod_filepath, output_folder=pyccel_dirpath.replace('/','.'), show_traceback=verbose)
        ast = parser.parse()
    except Exception:
        handle_error('parsing (syntax)')
        raise

    if syntax_only:
        return

    # Annotate abstract syntax Tree
    try:
        settings = {}
        ast = parser.annotate(**settings)
    except Exception:
        handle_error('annotation (semantic)')
        raise

    if semantic_only:
        return

    # Generate .f90 file
    try:
        codegen = Codegen(ast, module_name)
        fname = os.path.join(pyccel_dirpath, module_name)
        fname = codegen.export(fname)
    except Exception:
        handle_error('code generation')
        raise

    #------------------------------------------------------
    # TODO: collect dependencies and proceed recursively
#    if recursive:
#        for dep in parser.sons:
#            # Call same function on 'dep'
#            pass
    #------------------------------------------------------

    if convert_only:
        return

    # Reset Errors singleton
    errors = Errors()
    errors.reset()

    # Determine all .o files and all folders needed by executable
    def get_module_dependencies(parser, mods=(), folders=()):
        mod_folder = os.path.dirname(parser.filename) + "/__pyccel__/"
        mod_base = os.path.splitext(os.path.basename(parser.filename))[0]

        # Stop conditions
        if parser.metavars.get('ignore_at_import', False) or \
           parser.metavars.get('module_name', None) == 'omp_lib':
            return mods, folders

        # Update lists
        mods = [*mods, mod_folder + mod_base]
        folders = [*folders, mod_folder]

        # Proceed recursively
        for son in parser.sons:
            mods, folders = get_module_dependencies(son, mods, folders)

        return mods, folders

    dep_mods, inc_folders = get_module_dependencies(parser)
    includes += inc_folders

    if codegen.is_program:
        modules += [os.path.join(pyccel_dirpath, m) for m in dep_mods[1:]]


    # Construct compiler flags
    flags = construct_flags(f90exec,
                            fflags=fflags,
                            debug=debug,
                            accelerator=accelerator,
                            includes=includes,
                            libdirs=libdirs)

    # Compile Fortran code
    #
    # TODO: stop at object files, do not compile executable
    #       This allows for properly linking program to modules
    #
    try:
        compile_fortran(fname, f90exec, flags,
                        binary=None,
                        verbose=verbose,
                        modules=modules,
                        is_module=codegen.is_module,
                        output=pyccel_dirpath,
                        libs=libs)
    except Exception:
        handle_error('Fortran compilation')
        raise

    # For a program stop here
    if codegen.is_program:
        if verbose:
            exec_filepath = os.path.join(folder, module_name)
            print( '> Executable has been created: {}'.format(exec_filepath))
        os.chdir(base_dirpath)
        return

    # Create shared library
    try:
        sharedlib_filepath = create_shared_library(codegen,
                                                   pyccel_dirpath,
                                                   compiler,
                                                   mpi_compiler,
                                                   accelerator,
                                                   dep_mods,
                                                   extra_args,
                                                   output_name)
    except Exception:
        handle_error('shared library generation')
        raise

    # Move shared library to folder directory
    # (First construct absolute path of target location)
    sharedlib_filename = os.path.basename(sharedlib_filepath)
    target = os.path.join(folder, sharedlib_filename)
    shutil.move(sharedlib_filepath, target)
    sharedlib_filepath = target

    # Change working directory back to starting point
    os.chdir(base_dirpath)

    if verbose:
        print( '> Shared library has been created: {}'.format(sharedlib_filepath))
