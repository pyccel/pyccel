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
def execute_pyccel(fname, *,
                   syntax_only   = False,
                   semantic_only = False,
                   convert_only  = False,
                   recursive     = False,
                   verbose       = False,
                   folder        = None,
                   compiler    = None,
                   fflags      = None,
                   include     = [],
                   libdir      = [],
                   modules     = [],
                   libs        = [],
                   debug       = False,
                   extra_args  = '',
                   accelerator = None,
                   mpi         = False):

    # Store current directory
    base_dirpath = os.getcwd()

    # Handle any exception by printing error message, moving to original
    # working directory, and then raising the exception to the caller:
    def raise_error(stage):
        print('\nERROR at {} stage'.format(stage))
        os.chdir(base_dirpath)
        raise

    # Identify absolute path, directory, and filename
    pymod_filepath = os.path.abspath(fname)
    pymod_dirpath, pymod_filename = os.path.split(pymod_filepath)

    # Extract module name
    module_name = os.path.splitext(pymod_filename)[0]

    # Define working directory 'folder'
    if folder is None:
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

    f90exec = 'mpif90' if mpi else compiler

    # ...
    # Construct flags for the Fortran compiler
    if fflags is None:
        fflags = construct_flags(f90exec,
                                 fflags=None,
                                 debug=debug,
                                 accelerator=accelerator,
                                 include=[],
                                 libdir=[])

    # Build position-independent code, suited for use in shared library
    fflags = ' {} -fPIC '.format(fflags)
    # ...

    # Parse Python file
    try:
        parser = Parser(pymod_filepath, output_folder=pyccel_dirpath.replace('/','.'))
        ast = parser.parse()
    except:
        raise_error('parsing (syntax)')

    if syntax_only:
        return

    # Annotate abstract syntax Tree
    try:
        settings = {}
        ast = parser.annotate(**settings)
    except:
        raise_error('annotation (semantic)')

    if semantic_only:
        return

    # Generate .f90 file
    try:
        codegen = Codegen(ast, module_name)
        fname = os.path.join(pyccel_dirpath, module_name)
        fname = codegen.export(fname)
    except:
        raise_error('code generation')

    #------------------------------------------------------
    # TODO: collect dependencies and proceed recursively
    if recursive:
        for dep in parser.sons:
            # Call same function on 'dep'
            pass
    #------------------------------------------------------

    if convert_only:
        return

    # Reset Errors singleton
    errors = Errors()
    errors.reset()

    # Construct compiler flags
    flags = construct_flags(f90exec,
                            fflags=fflags,
                            debug=debug,
                            accelerator=accelerator,
                            include=include,
                            libdir=libdir)

    # Compile Fortran code
    #
    # TODO: stop at object files, do not compile executable
    #       This allows for properly linking program to modules
    #
    try:

        # Determine all .o files needed by executable
        if codegen.is_program:
            def get_module_dependencies(parser, mods=[]):
                mods = mods + [os.path.splitext(os.path.basename(parser.filename))[0]]
                for son in parser.sons:
                    mods = get_module_dependencies(son, mods)
                return mods
            dep_mods = get_module_dependencies(parser)[1:] # NOTE: avoid parent
            modules += [os.path.join(pyccel_dirpath, m) for m in dep_mods]

        output, cmd = compile_fortran(fname, f90exec, flags,
                                      binary=None,
                                      verbose=verbose,
                                      modules=modules,
                                      is_module=codegen.is_module,
                                      output=pyccel_dirpath,
                                      libs=libs)
    except:
        raise_error('Fortran compilation')

    # For a program stop here
    if codegen.is_program:
        if verbose:
            exec_filepath = os.path.join(folder, module_name)
            print( '> Executable has been created: {}'.format(exec_filepath))
        os.chdir(base_dirpath)
        return

    # Create shared library
    try:
        sharedlib_filepath = create_shared_library(parser,
                                                   codegen,
                                                   pyccel_dirpath,
                                                   compiler,
                                                   accelerator,
                                                   mpi,
                                                   extra_args)
    except:
        raise_error('shared library generation')

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
