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
from pyccel.errors.errors import PyccelError

__all__ = ['random_string', 'get_source_function', 'epyccel_seq', 'epyccel']

#==============================================================================
random_selector = random.SystemRandom()

def random_string( n ):
    # we remove uppercase letters because of f2py
    chars    = string.ascii_lowercase + string.digits
    return ''.join( random_selector.choice( chars ) for _ in range(n) )

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
def epyccel_seq(function_or_module, *,
                language     = None,
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

        while module_name in sys.modules.keys():
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
        module_import_prefix = pymod.__name__ + '_'
        while module_import_prefix + tag in sys.modules.keys():
            tag = random_string(n=8)

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
                       language    = language,
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

    # http://ballingt.com/import-invalidate-caches
    # https://docs.python.org/3/library/importlib.html#importlib.invalidate_caches
    importlib.invalidate_caches()

    package = importlib.import_module(module_name)
    sys.path.remove(epyccel_dirpath)

    # Verify that we have imported the shared library, not the Python one
    loader = getattr(package, '__loader__', None)
    if not isinstance(loader, ExtensionFileLoader):
        raise ImportError('Could not load shared library')

    # If Python object was function, extract it from module
    if isinstance(function_or_module, FunctionType):
        func = getattr(package, pyfunc.__name__.lower())
    else:
        func = None

    # Return accelerated Python module and function
    return package, func

#==============================================================================
def epyccel( python_function_or_module, **kwargs ):
    """
    Accelerate Python function or module using Pyccel in "embedded" mode.

    Parameters
    ----------
    python_function_or_module : function | module
        Python function or module to be accelerated.

    verbose : bool
        Print additional information (default: False).

    language : {'fortran', 'c', 'python'}
        Language of generated code (default: 'fortran').

    accelerator : str, optional
        Parallel multi-threading acceleration strategy
        (currently supported: 'openmp', 'openacc').

    Options for parallel mode
    -------------------------
    comm : mpi4py.MPI.Comm, optional
        MPI communicator for calling Pyccel in parallel mode (default: None).

    root : int, optional
        MPI rank of process in charge of accelerating code (default: 0).

    bcast : {True, False}
        If False, only root process loads accelerated function/module (default: True).

    Other options
    -------------
    compiler : str, optional
        User-defined command for compiling generated source code.

    mpi_compiler : str, optional
        Compiler for MPI parallel code.

    Returns
    -------
    res : object
        Accelerated function or module.

    Examples
    --------
    >>> def one(): return 1
    >>> from pyccel.epyccel import epyccel
    >>> one_f = epyccel(one, language='fortran')
    >>> one_c = epyccel(one, language='c')

    """
    assert isinstance( python_function_or_module, (FunctionType, ModuleType) )

    comm  = kwargs.pop('comm', None)
    root  = kwargs.pop('root', 0)
    bcast = kwargs.pop('bcast', True)

    # Parallel version
    if comm is not None:

        from mpi4py import MPI
        from tblib  import pickling_support   # [YG, 27.10.2020] We use tblib to
        pickling_support.install()            # pickle tracebacks, which allows
                                              # mpi4py to broadcast exceptions
        assert isinstance( comm, MPI.Comm )
        assert isinstance( root, int      )

        # TODO [YG, 25.02.2020] Get default MPI compiler from somewhere else
        kwargs.setdefault('mpi_compiler', 'mpif90')

        # Master process calls epyccel
        if comm.rank == root:
            try:
                mod, fun = epyccel_seq( python_function_or_module, **kwargs )
                mod_path = os.path.abspath(mod.__file__)
                mod_name = mod.__name__
                fun_name = python_function_or_module.__name__ if fun else None
                success  = True
            # error handling carried out after broadcast to prevent deadlocks
            except: # pylint: disable=bare-except
                exc_info = sys.exc_info()
                success  = False

        # Non-master processes initialize empty variables
        else:
            mod, fun = None, None
            mod_path = None
            mod_name = None
            fun_name = None
            exc_info = None
            success  = None

        # Broadcast success state, and raise exception if neeeded
        if not comm.bcast(success, root=root):
            raise comm.bcast(exc_info, root=root)

        if bcast:
            # Broadcast Fortran module path/name and function name to all processes
            mod_path = comm.bcast( mod_path, root=root )
            mod_name = comm.bcast( mod_name, root=root )
            fun_name = comm.bcast( fun_name, root=root )

            # Non-master processes import Fortran module directly from its path
            # and extract function if its name is given
            if comm.rank != root:
                folder = os.path.split(mod_path)[0]
                sys.path.insert(0, folder)
                mod = importlib.import_module(mod_name)
                sys.path.remove(folder)
                fun = getattr(mod, fun_name) if fun_name else None

    # Serial version
    else:
        mod, fun = epyccel_seq( python_function_or_module, **kwargs )

    # Return Fortran function (if any), otherwise module
    return fun or mod
