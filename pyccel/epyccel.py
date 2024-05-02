# coding: utf-8

import inspect
import importlib
import sys
import os
import string
import random

from types import ModuleType, FunctionType
from importlib.machinery import ExtensionFileLoader

from pyccel.codegen.pipeline import execute_pyccel

__all__ = ['random_string', 'get_source_function', 'epyccel_seq', 'epyccel']

#==============================================================================
def random_string( n ):
    # we remove uppercase letters because of f2py
    chars    = string.ascii_lowercase + string.digits
    selector = random.SystemRandom()
    return ''.join( selector.choice( chars ) for _ in range( n ) )

#==============================================================================
def get_source_function(func):
    if not callable(func):
        raise TypeError('Expecting a callable function')

    lines = inspect.getsourcelines(func)
    lines = lines[0]
    # remove indentation if the first line is indented
    a = lines[0]
    leading_spaces = len(a) - len(a.lstrip())
    code = ''
    for a in lines:
        if leading_spaces > 0:
            line = a[leading_spaces:]
        else:
            line = a
        code = '{code}{line}'.format(code=code, line=line)

    return code

#==============================================================================
def epyccel_seq(function_or_module,
                compiler     = None,
                mpi_compiler = None,
                fflags       = None,
                accelerator  = None,
                verbose      = False,
                debug        = False,
                includes     = (),
                libdirs      = (),
                modules      = (),
                libs         = (),
                extra_args   = '',
                folder       = None):

    # ... get the module source code
    if isinstance(function_or_module, FunctionType):
        pyfunc = function_or_module
        code = get_source_function(pyfunc)
        tag = random_string(8)
        module_name = 'mod_{}'.format(tag)
        pymod_filename = '{}.py'.format(module_name)
        pymod_filepath = os.path.abspath(pymod_filename)

    elif isinstance(function_or_module, ModuleType):
        pymod = function_or_module
        pymod_filepath = pymod.__file__
        pymod_filename = os.path.basename(pymod_filepath)
        lines = inspect.getsourcelines(pymod)[0]
        code = ''.join(lines)
        tag = random_string(8)
        module_name = pymod.__name__.split('.')[-1] + '_' + tag

    else:
        raise TypeError('> Expecting a FunctionType or a ModuleType')
    # ...

    # Store current directory
    base_dirpath = os.getcwd()

    # Define working directory 'folder'
    if folder is None:
        folder = os.path.dirname(pymod_filepath)
    else:
        folder = os.path.abspath(folder)

    # Define directory name and path for epyccel files
    epyccel_dirname = '__epyccel__'
    epyccel_dirpath = os.path.join(folder, epyccel_dirname)

    # Create new directories if not existing
    os.makedirs(folder, exist_ok=True)
    os.makedirs(epyccel_dirpath, exist_ok=True)

    # Change working directory to '__epyccel__'
    os.chdir(epyccel_dirpath)

    # Store python file in '__epyccel__' folder, so that execute_pyccel can run
    with open(pymod_filename, 'w') as f:
        f.writelines(code)

    try:
        # Generate shared library
        execute_pyccel(pymod_filename,
                       verbose     = verbose,
                       compiler    = compiler,
                       mpi_compiler= mpi_compiler,
                       fflags      = fflags,
                       includes    = includes,
                       libdirs     = libdirs,
                       modules     = modules,
                       libs        = libs,
                       debug       = debug,
                       extra_args  = extra_args,
                       accelerator = accelerator,
                       output_name = module_name)
    finally:
        # Change working directory back to starting point
        os.chdir(base_dirpath)

    # Import shared library
    sys.path.insert(0, epyccel_dirpath)
    package = importlib.import_module(module_name)
    sys.path.remove(epyccel_dirpath)

    # Verify that we have imported the shared library, not the Python one
    loader = getattr(package, '__loader__', None)
    if not isinstance(loader, ExtensionFileLoader):
        raise ImportError('Could not load shared library')

    # Function case:
    if isinstance(function_or_module, FunctionType):
        return getattr(package, pyfunc.__name__.lower())

    # Module case:
    return package

#==============================================================================
def epyccel( inputs, **kwargs ):

    comm = kwargs.pop('comm', None)
    root = kwargs.pop('root', 0)
    bcast = kwargs.pop('bcast', True)

    if comm is not None:
        # TODO not tested for a function
        from mpi4py import MPI

        assert isinstance( comm, MPI.Comm )
        assert isinstance( root, int      )

        # TODO [YG, 25.02.2020] Get default MPI compiler from somewhere else
        kwargs.setdefault('mpi_compiler', 'mpif90')

        # Master process calls epyccel
        if comm.rank == root:
            fmod      = epyccel_seq( inputs, **kwargs )
            fmod_path = os.path.abspath(fmod.__file__)
            fmod_name = fmod.__name__
        else:
            fmod      = None
            fmod_path = None
            fmod_name = None

        if bcast:
            # Broadcast Fortran module path/name to all processes
            fmod_path = comm.bcast( fmod_path, root=root )
            fmod_name = comm.bcast( fmod_name, root=root )

            # Non-master processes import Fortran module directly from its path
            if comm.rank != root:
                folder = os.path.split(fmod_path)[0]
                sys.path.insert(0, folder)
                fmod = importlib.import_module(fmod_name)
                sys.path.remove(folder)

        # Return Fortran module
        return fmod

    else:
        return epyccel_seq( inputs, **kwargs )
