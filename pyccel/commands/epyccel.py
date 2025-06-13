# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
""" File containing functions for calling Pyccel interactively (epyccel and epyccel_seq)
"""

import inspect
import importlib
import re
import sys
import typing
import os

from filelock import FileLock, Timeout

from types import ModuleType, FunctionType
from importlib.machinery import ExtensionFileLoader

from pyccel.utilities.strings  import random_string
from pyccel.codegen.pipeline   import execute_pyccel
from pyccel.errors.errors      import ErrorsMode, PyccelError, Errors

errors = Errors()

__all__ = ['get_source_code_and_context', 'epyccel_seq', 'epyccel']


#==============================================================================
def get_source_code_and_context(func_or_class):
    """
    Get the source code and context from a function or a class.

    Get a string containing the source code of a function from a function
    object or the source code of a class from a class object. Excessive
    indenting is stripped away.

    Additionally retrieve the information about the variables that are
    available in the calling context. This can allow certain constants such as
    type hints to be defined outside of the function passed to epyccel.

    Parameters
    ----------
    func_or_class : Function | type
        A Python function or class.

    Returns
    -------
    code : list[str]
        A list of strings containing the lines of the source code.
    context_dict : dict[str, object]
        A dictionary containing any objects defined in the context
        which may be useful for the function.

    Raises
    ------
    TypeError
        A type error is raised if the object passed to the function is not
        callable.
    """
    if not callable(func_or_class):
        raise TypeError('Expecting a callable function')

    lines, _ = inspect.getsourcelines(func_or_class)
    # remove indentation if the first line is indented
    unindented_line = lines[0]
    leading_spaces = len(unindented_line) - len(unindented_line.lstrip())
    lines = [l[leading_spaces:] for l in lines]

    # Strip trailing comments (e.g. pylint disable)
    commentless_lines = []
    for l in lines:
        comment = l.rfind('#')
        # Avoid # in a string
        if l[:comment].count("'") % 2 == 0 and l[:comment].count("'") % 2 == 0:
            l = l[:comment] + '\n'
        commentless_lines.append(l)

    # Search for methods
    methods = [(func_or_class.__name__, func_or_class)] if isinstance(func_or_class, FunctionType) else \
                inspect.getmembers(func_or_class, predicate=inspect.isfunction)

    func_name_regex = re.compile(r'^\s*def\s+([a-zA-Z0-9_]+)\s*\(')
    func_match = [re.match(func_name_regex, l) for l in lines]
    prototypes = {m[1]: i for i, m in enumerate(func_match) if m}

    if len(methods) == 0:
        return ''.join(lines), {}

    # Build context dict with globals
    _, method0 = methods[0]
    context_dict = method0.__globals__.copy()

    # Sort the methods in reverse order of appearance to preserve line indices
    # during treatment of methods
    methods.sort(key = lambda m: prototypes[m[0]], reverse=True)

    for m_name, m in methods:
        # Update context dict with closure vars
        context_dict.update(inspect.getclosurevars(m).nonlocals)

        # Print signature (Python has already executed the line with the prototype
        # so any variables in the line cannot be deduced from the closure vars or
        # globals, the only way to get a clean version is to reprint the signature)
        sig = inspect.signature(m)
        prototype_idx = prototypes[m_name]

        method_prototype = lines[prototype_idx]
        indent = len(method_prototype) - len(method_prototype.lstrip())

        # Handle multi-line prototypes
        end_of_prototype_idx = next(i for i, l in enumerate(commentless_lines[prototype_idx:]) if l.strip().endswith(':'))
        if end_of_prototype_idx > prototype_idx:
            lines = lines[:prototype_idx+1] + lines[end_of_prototype_idx+1:]

        method_prototype = ' '*indent + f'def {m_name}{sig}:\n'

        # TypeVar in a signature appear as +T, -T or ~T but the associated variable
        # T will not be available from the closure vars or globals, we therefore
        # search for TypeVars, put their definition in the context_dict and use the
        # variable name (e.g. T) in the signature.
        params = {p.annotation for p in sig.parameters.values()}
        while params:
            annot = params.pop()
            if isinstance(annot, typing.TypeVar):
                name = annot.__name__
                if name in context_dict:
                    if context_dict[name] != annot:
                        errors.report("Multiple TypeVars found with the same name",
                                severity='fatal', line=1)
                else:
                    context_dict[name] = annot
                method_prototype = method_prototype.replace(str(annot), name)
            elif isinstance(annot, typing.GenericAlias) or getattr(annot, '__origin__', None) is typing.Final:
                params.update(typing.get_args(annot))

        # Save the updated prototype
        lines[prototype_idx] = method_prototype

    return ''.join(lines), context_dict

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
def epyccel_seq(function_class_or_module, *,
                language      = None,
                compiler      = None,
                fflags        = None,
                wrapper_flags = None,
                accelerators  = (),
                verbose       = False,
                time_execution  = False,
                debug         = None,
                includes      = (),
                libdirs       = (),
                modules       = (),
                libs          = (),
                folder        = None,
                conda_warnings= 'basic',
                context_dict  = None,
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
    function_class_or_module : function | class | module | str
        Python function or module to be accelerated.
        If a string is passed then it is assumed to be the code from a module which
        should be accelerated. The module must be capable of running as a standalone
        file so it must include any necessary import statements.
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
    time_execution : bool
        Time the execution of Pyccel's internal stages.
    debug : bool, optional
        Enable debug mode. The default value is taken from the environment variable PYCCEL_DEBUG_MODE.
        If no such environment variable exists then the default is False.
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
    context_dict : dict[str, obj], optional
        A dictionary containing any Python objects from the calling scope which should
        be made available to the translated code. By default any objects that are used
        in the body of the function are made available, as well as any global objects.
        If the argument is provided then these objects will be treated as additional
        to the default arguments.

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

    if isinstance(function_class_or_module, (FunctionType, type, str)):
        dirpath = os.getcwd()

    elif isinstance(function_class_or_module, ModuleType):
        dirpath = os.path.dirname(function_class_or_module.__file__)

    # Define working directory 'folder'
    if folder is None:
        folder = dirpath
    else:
        folder = os.path.abspath(folder)

    # Define directory name and path for epyccel files
    epyccel_dirname = '__epyccel__' + os.environ.get('PYTEST_XDIST_WORKER', '')
    epyccel_dirpath = os.path.join(folder, epyccel_dirname)

    # ... get the module source code
    if isinstance(function_class_or_module, (FunctionType, type)):
        code, collected_context_dict = get_source_code_and_context(function_class_or_module)
        if context_dict:
            collected_context_dict.update(context_dict)
        context_dict = collected_context_dict

        module_name, module_lock = get_unique_name('mod', epyccel_dirpath)

    elif isinstance(function_class_or_module, ModuleType):
        pymod = function_class_or_module
        lines = inspect.getsourcelines(pymod)[0]
        code = ''.join(lines)

        module_name, module_lock = get_unique_name(pymod.__name__, epyccel_dirpath)

    elif isinstance(function_class_or_module, str):
        code = function_class_or_module

        module_name, module_lock = get_unique_name('mod', epyccel_dirpath)

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
                           verbose         = verbose,
                           show_timings    = time_execution,
                           language        = language,
                           compiler_family = compiler,
                           fflags          = fflags,
                           wrapper_flags   = wrapper_flags,
                           includes        = includes,
                           libdirs         = libdirs,
                           modules         = modules,
                           libs            = libs,
                           debug           = debug,
                           accelerators    = accelerators,
                           output_name     = module_name,
                           conda_warnings  = conda_warnings,
                           context_dict    = context_dict)
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

        if language and language.lower() != 'python':
            # Verify that we have imported the shared library, not the Python one
            loader = getattr(package, '__loader__', None)
            if not isinstance(loader, ExtensionFileLoader):
                raise ImportError('Could not load shared library')

        # If Python object was a function or a class, extract it from module
        if isinstance(function_class_or_module, (FunctionType, type)):
            func = getattr(package, function_class_or_module.__name__)
        else:
            func = None
    finally:
        module_lock.release()

    # Return accelerated Python module and function
    return package, func

#==============================================================================
def epyccel( python_function_class_or_module, **kwargs ):
    """
    Accelerate Python function or module using Pyccel in "embedded" mode.

    This function accelerates a Python function or module using Pyccel in "embedded" mode.
    It generates optimized code in the specified language (default is 'fortran')
    and compiles it for improved performance

    Parameters
    ----------
    python_function_class_or_module : function | class | module | str
        Python function or module to be accelerated.
        If a string is passed then it is assumed to be the code from a module which
        should be accelerated.
    **kwargs :
        Additional keyword arguments for configuring the compilation and acceleration process.
        Available options are defined in epyccel_seq.

    Returns
    -------
    object
        Accelerated function, class or module.

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
    assert isinstance( python_function_class_or_module, (FunctionType, type, ModuleType, str) )

    comm  = kwargs.pop('comm', None)
    root  = kwargs.pop('root', 0)
    bcast = kwargs.pop('bcast', True)
    # This will initialize the singleton ErrorsMode
    # making this setting available everywhere
    err_mode = ErrorsMode()
    if kwargs.pop('developer_mode', None):
        err_mode.set_mode('developer')
    else:
        err_mode.set_mode(os.environ.get('PYCCEL_ERROR_MODE', 'user'))

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
                mod, obj = epyccel_seq( python_function_class_or_module, **kwargs )
                mod_path = os.path.abspath(mod.__file__)
                mod_name = mod.__name__
                obj_name = python_function_class_or_module.__name__ if obj else None
                success  = True
            # error handling carried out after broadcast to prevent deadlocks
            except PyccelError as e:
                raise type(e)(str(e)) from None
            except BaseException as e: # pylint: disable=broad-except
                exc_info = e
                success  = False

        # Non-master processes initialize empty variables
        else:
            mod, obj = None, None
            mod_path = None
            mod_name = None
            obj_name = None
            exc_info = None
            success  = None

        # Broadcast success state, and raise exception if needed
        if not comm.bcast(success, root=root):
            raise comm.bcast(exc_info, root=root)

        if bcast:
            # Broadcast Fortran module path/name and function name to all processes
            mod_path = comm.bcast( mod_path, root=root )
            mod_name = comm.bcast( mod_name, root=root )
            obj_name = comm.bcast( obj_name, root=root )

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
                obj = getattr(mod, obj_name) if obj_name else None

    # Serial version
    else:
        try:
            mod, obj = epyccel_seq( python_function_class_or_module, **kwargs )
        except PyccelError as e:
            raise type(e)(str(e)) from None

    # Return Fortran function (if any), otherwise module
    return obj or mod
