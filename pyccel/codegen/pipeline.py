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
# TODO: prune options
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

    #------------------------------------------------------
    # NOTE:
    # [..]_dirname is the name of a directory
    # [..]_dirpath is the full (absolute) path of a directory
    #------------------------------------------------------

    # Store current directory
    base_dirpath = os.getcwd()

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
        if mpi == True:
            compiler = 'mpif90'
        else:
            compiler = 'gfortran'

    # ...
    # Construct flags for the Fortran compiler
    if fflags is None:
        fflags = construct_flags(compiler,
                                 fflags=None,
                                 debug=debug,
                                 accelerator=accelerator,
                                 include=[],
                                 libdir=[])

    # Build position-independent code, suited for use in shared library
    fflags = ' {} -fPIC '.format(fflags)
    # ...

    # Parse Python file
    parser = Parser(pymod_filepath, output_folder=pyccel_dirpath.replace('/','.'))
    ast = parser.parse()

    if syntax_only:
        return

    # Annotate abstract syntax Tree
    settings = {}
    ast = parser.annotate(**settings)

    if semantic_only:
        return

    # Generate .f90 file
    codegen = Codegen(ast, module_name)
    fname = os.path.join(pyccel_dirpath, module_name)
    fname = codegen.export(fname)

    # TODO: collect dependencies and proceed recursively
    if recursive:
        for dep in parser.sons:
            # Call same function on 'dep'
            pass

    if convert_only:
        return

    # Reset Errors singleton
    errors = Errors()
    errors.reset()

    # Construct compiler flags
    if compiler is None:
        compiler = 'gfortran'

    flags = construct_flags(compiler,
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
    output, cmd = compile_fortran(fname, compiler, flags,
                                  binary=None,
                                  verbose=verbose,
                                  modules=modules,
                                  is_module=codegen.is_module,
                                  output=pyccel_dirpath,
                                  libs=libs)

    # For a program stop here
    if codegen.is_program:
        return

    # Create shared library
    sharedlib_filepath = create_shared_library(parser,
                                               codegen,
                                               pyccel_dirpath,
                                               compiler,
                                               accelerator,
                                               mpi,
                                               extra_args)

    # Move shared library to folder directory
    # (First construct absolute path of target location)
    sharedlib_filename = os.path.basename(sharedlib_filepath)
    new_sharedlib_filepath = os.path.join(folder, sharedlib_filename)
    shutil.move(sharedlib_filepath, new_sharedlib_filepath)

    # Change working directory back to starting point
    os.chdir(base_dirpath)

    if verbose:
        print( '> Shared library has been created: {}'.format(sharedlib_filepath))
