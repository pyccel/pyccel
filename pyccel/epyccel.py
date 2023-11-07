# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#


import inspect
import importlib
import sys
import os
import string
import random
from filelock import FileLock, Timeout

from types import ModuleType, FunctionType
from importlib.machinery import ExtensionFileLoader

from pyccel.codegen.pipeline import execute_pyccel
from pyccel.errors.errors import ErrorsMode

__all__ = ['random_string', 'get_source_function', 'epyccel_seq', 'epyccel']

#==============================================================================
random_selector = random.SystemRandom()

def random_string( n ):
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
def get_unique_name(prefix, path):
    """
    Get a unique module name.

    Get a unique name based on the prefix which does not coincide with a
    module which already exists and is not being created by another thread.

    Parameters
    ----------
    prefix : str
        The starting string of the random name.
    path : str
        The folder where the lock file should be saved.

    Returns
    -------
    module_name : str
                  A unique name for the new module.
    module_lock : FileLock
                  A file lock preventing other threads
                  from creating a module with the same name.
    """
    module_import_prefix = prefix + '_'

    # Find an unused name
    tag = random_string(12)
    module_name = module_import_prefix + tag

    while module_name in sys.modules.keys():
        tag = random_string(12)
        module_name = module_import_prefix + tag

    module_name = module_name.split('.')[-1]

    # Create new directories if not existing
    os.makedirs(path, exist_ok=True)

    # Ensure that the name is not in use by another thread
    lock = FileLock(os.path.join(path, module_name) + '.lock')
    try:
        lock.acquire(timeout=0.1)
        if module_name in sys.modules.keys():
            raise Timeout("Newly created collision")
    except Timeout:
        return get_unique_name(prefix, path)
    return module_name, lock

#==============================================================================
def epyccel_seq(function_or_module, *,
                language      = None,
                compiler      = None,
                fflags        = None,
                wrapper_flags = None,
                accelerators  = (),
                verbose       = False,
                debug         = False,
                includes      = (),
                libdirs       = (),
                modules       = (),
                libs          = (),
                folder        = None,
                conda_warnings= 'basic',
                comm          = None,
                root          = None,
                bcast         = None):
    """
    Accelerate Python function or module using Pyccel in "embedded" mode.

    This function accelerates a Python function or module using Pyccel in "embedded" mode.
    It generates optimized code in the specified language (default is 'fortran')
    and compiles it for improved performance.

    Parameters
    ----------
    function_or_module : function | module
        Python function or module to be accelerated.
    language : {'fortran', 'c', 'python'}
        Language of generated code (default: 'fortran').
    compiler : str, optional
        User-defined command for compiling generated source code.
    fflags : iterable of str, optional
        Compiler flags.
    wrapper_flags : iterable of str, optional
        Flags to be passed to the wrapper code generator.
    accelerators : iterable of str, optional
        Parallel multi-threading acceleration strategy
        (currently supported: 'mpi', 'openmp', 'openacc').
    verbose : bool
        Print additional information (default: False).
    debug : bool, optional
        Enable debug mode.
    includes : tuple, optional
        Additional include directories for the compiler.
    libdirs : tuple, optional
        Additional library directories for the compiler.
    modules : tuple, optional
        Additional modules to be imported.
    libs : tuple, optional
        Additional libraries.
    folder : str, optional
        Output folder for the compiled code.
    conda_warnings : {off, basic, verbose}
        Specify the level of Conda warnings to display (choices: off, basic, verbose), Default is 'basic'.

    Returns
    -------
    object
        Return accelerated Python module and function.

    Other Parameters
    ----------------
    comm : mpi4py.MPI.Comm, optional
        MPI communicator for calling Pyccel in parallel mode (default: None) (for parallel mode).
    root : int, optional
        MPI rank of process in charge of accelerating code (default: 0) (for parallel mode).
    bcast : {True, False}
        If False, only root process loads accelerated function/module (default: True) (for parallel mode). 
    """
    # Store current directory
    base_dirpath = os.getcwd()

    if isinstance(function_or_module, (FunctionType, type)):
        dirpath = os.getcwd()

    elif isinstance(function_or_module, ModuleType):
        dirpath = os.path.dirname(function_or_module.__file__)

    # Define working directory 'folder'
    if folder is None:
        folder = os.path.dirname(dirpath)
    else:
        folder = os.path.abspath(folder)

    # Define directory name and path for epyccel files
    epyccel_dirname = '__epyccel__' + os.environ.get('PYTEST_XDIST_WORKER', '')
    epyccel_dirpath = os.path.join(folder, epyccel_dirname)

    # ... get the module source code
    if isinstance(function_or_module, (FunctionType, type)):
        pyfunc = function_or_module
        code = get_source_function(pyfunc)

        module_name, module_lock = get_unique_name('mod', epyccel_dirpath)

    elif isinstance(function_or_module, ModuleType):
        pymod = function_or_module
        lines = inspect.getsourcelines(pymod)[0]
        code = ''.join(lines)

        module_name, module_lock = get_unique_name(pymod.__name__, epyccel_dirpath)

    else:
        raise TypeError('> Expecting a FunctionType, type or a ModuleType')

    # Try is necessary to ensure lock is released
    try:
        pymod_filename = '{}.py'.format(module_name)
        pymod_filepath = os.path.join(dirpath, pymod_filename)
        # ...

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
                           verbose       = verbose,
                           language      = language,
                           compiler      = compiler,
                           fflags        = fflags,
                           wrapper_flags = wrapper_flags,
                           includes      = includes,
                           libdirs       = libdirs,
                           modules       = modules,
                           libs          = libs,
                           debug         = debug,
                           accelerators  = accelerators,
                           output_name   = module_name,
                           conda_warnings= conda_warnings)
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

        if language != 'python':
            # Verify that we have imported the shared library, not the Python one
            loader = getattr(package, '__loader__', None)
            if not isinstance(loader, ExtensionFileLoader):
                raise ImportError('Could not load shared library')

        # If Python object was function, extract it from module
        if isinstance(function_or_module, (FunctionType, type)):
            func = getattr(package, pyfunc.__name__)
        else:
            func = None
    finally:
        module_lock.release()

    # Return accelerated Python module and function
    return package, func

#==============================================================================
def epyccel( python_function_or_module, **kwargs ):
    """
    Accelerate Python function or module using Pyccel in "embedded" mode.

    This function accelerates a Python function or module using Pyccel in "embedded" mode.
    It generates optimized code in the specified language (default is 'fortran')
    and compiles it for improved performance

    Parameters
    ----------
    python_function_or_module : function | module
        Python function or module to be accelerated.
    **kwargs :
        Additional keyword arguments for configuring the compilation and acceleration process.
        Available options are defined in epyccel_seq.

    Returns
    -------
    object
        Accelerated function or module.

    See Also 
    -------- 
    epyccel_seq
        The version of this function called in a sequential context.

    Examples
    --------
    >>> def one(): return 1
    >>> from pyccel.epyccel import epyccel
    >>> one_f = epyccel(one, language='fortran')
    >>> one_c = epyccel(one, language='c')
    """
    assert isinstance( python_function_or_module, (FunctionType, type, ModuleType) )

    comm  = kwargs.pop('comm', None)
    root  = kwargs.pop('root', 0)
    bcast = kwargs.pop('bcast', True)
    if kwargs.pop('developer_mode', None):
        # This will initialize the singleton ErrorsMode
        # making this setting available everywhere
        err_mode = ErrorsMode()
        err_mode.set_mode('developer')

    # Parallel version
    if comm is not None:

        from mpi4py import MPI
        from tblib  import pickling_support   # [YG, 27.10.2020] We use tblib to
        pickling_support.install()            # pickle tracebacks, which allows
                                              # mpi4py to broadcast exceptions
        assert isinstance( comm, MPI.Comm )
        assert isinstance( root, int      )

        kwargs.setdefault('accelerators', [])
        if 'mpi' not in kwargs['accelerators']:
            kwargs['accelerators'] = [*kwargs['accelerators'], 'mpi']

        # Master process calls epyccel
        if comm.rank == root:
            try:
                mod, fun = epyccel_seq( python_function_or_module, **kwargs )
                mod_path = os.path.abspath(mod.__file__)
                mod_name = mod.__name__
                fun_name = python_function_or_module.__name__ if fun else None
                success  = True
            # error handling carried out after broadcast to prevent deadlocks
            except BaseException as e: # pylint: disable=broad-except
                exc_info = e
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
                # http://ballingt.com/import-invalidate-caches
                # https://docs.python.org/3/library/importlib.html#importlib.invalidate_caches
                importlib.invalidate_caches()
                mod = importlib.import_module(mod_name)
                sys.path.remove(folder)
                fun = getattr(mod, fun_name) if fun_name else None

    # Serial version
    else:
        mod, fun = epyccel_seq( python_function_or_module, **kwargs )

    # Return Fortran function (if any), otherwise module
    return fun or mod
