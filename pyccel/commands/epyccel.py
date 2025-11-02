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
from types import ModuleType, FunctionType
from importlib.machinery import ExtensionFileLoader

from filelock import FileLock, Timeout

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

    while module_name in sys.modules:
        tag = random_string(12)
        module_name = module_import_prefix + tag

    module_name = module_name.split('.')[-1]

    # Create new directories if not existing
    os.makedirs(path, exist_ok=True)

    # Ensure that the name is not in use by another thread
    lock = FileLock(os.path.join(path, module_name) + '.lock')
    try:
        lock.acquire(timeout=0.1)
        if module_name in sys.modules:
            raise Timeout("Newly created collision")
    except Timeout:
        return get_unique_name(prefix, path)
    return module_name, lock

#==============================================================================
def epyccel_seq(function_class_or_module, *,
                language        = 'fortran',
                compiler_family = None,
                compiler_config = None,
                flags           = None,
                wrapper_flags   = None,
                debug           = None,
                include         = (),
                libdir          = (),
                libs            = (),
                folder          = None,
                mpi             = False,
                openmp          = False,
                openacc         = False,
                verbose         = 0,
                time_execution  = False,
                conda_warnings  = 'basic',
                context_dict    = None
    ):
    """
    Accelerate Python function or module using Pyccel in "embedded" mode.

    This function accelerates a Python function or module using Pyccel in "embedded" mode.
    It generates optimized code in the specified language (default is 'fortran')
    and compiles it for improved performance. Please be aware that only one of
    the parameters `compiler_family` and `compiler_config` may be provided.

    Parameters
    ----------
    function_class_or_module : function | class | module | str
        Python function, class, or module to be accelerated.
        If a string is passed then it is assumed to be the code from a module which
        should be accelerated. The module must be capable of running as a standalone
        file so it must include any necessary import statements.
    language : {'fortran', 'c', 'python'}
        Language of generated code (default: 'fortran').
    compiler_family : str, optional
        Compiler family for which Pyccel uses a default configuration (default: 'GNU').
    compiler_config : pathlib.Path | str, optional
        Path to a JSON file containing a compiler configuration (overrides compiler_family).
    flags : str, optional
        Compiler flags.
    wrapper_flags : str, optional
        Flags to be passed to the wrapper code generator.
    debug : bool, optional
        Whether the file should be compiled in debug mode.
        The default value is taken from the environment variable PYCCEL_DEBUG_MODE.
        If no such environment variable exists then the default is False.
    include : tuple, optional
        Additional include directories for the compiler.
    libdir : tuple, optional
        Additional library directories for the compiler.
    libs : tuple, optional
        Additional libraries.
    folder : str, optional
        Output folder for the compiled code.
    mpi : bool, default=False
        If True, use MPI for parallel execution.
    openmp : bool, default=False
        If True, use OpenMP for parallel execution.
    openacc : bool, default=False
        If True, use OpenACC for parallel execution.
    verbose : int, default=0
        Set the level of verbosity to see additional information about the Pyccel process.
    time_execution : bool
        Time the execution of Pyccel's internal stages.
    conda_warnings : {'off', 'basic', 'verbose'}
        Specify the level of Conda warnings to display (default: 'basic').
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
    """
    # Store current directory
    base_dirpath = os.getcwd()

    # Check if function_class_or_module is a valid type
    allowed_types = (FunctionType, type, str, ModuleType)
    if not isinstance(function_class_or_module, allowed_types):
        raise TypeError('> Expecting a FunctionType, type, str, or a ModuleType')

    # Check if compiler_family and compiler_config are mutually exclusive
    if None not in (compiler_family, compiler_config):
        raise TypeError('> Only one of the parameters `compiler_family` or `compiler_config` may be provided')

    # Get the directory path of the function or module
    if isinstance(function_class_or_module, (FunctionType, type, str)):
        dirpath = os.getcwd()
    else: # ModuleType
        dirpath = os.path.dirname(function_class_or_module.__file__)

    # Define working directory 'folder'
    folder = dirpath if folder is None else os.path.abspath(folder)

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

    elif isinstance(function_class_or_module, str):
        code = function_class_or_module
        module_name, module_lock = get_unique_name('mod', epyccel_dirpath)

    else: # ModuleType
        pymod = function_class_or_module
        lines = inspect.getsourcelines(pymod)[0]
        code = ''.join(lines)
        module_name, module_lock = get_unique_name(pymod.__name__, epyccel_dirpath)

    # execute_pyccel wants a unique string for the compiler family and the JSON file
    # The default compiler family is 'GNU' and is handled by execute_pyccel
    compiler_family_or_config = compiler_config or compiler_family
    # Convert from pathlib.Path to string if not None
    if compiler_family_or_config is not None:
        compiler_family_or_config = str(compiler_family_or_config)

    # Store the accelerators options into a tuple of strings
    accelerators = []
    if mpi:
        accelerators.append("mpi")
    if openmp:
        accelerators.append("openmp")
    if openacc:
        accelerators.append("openacc")
    accelerators = tuple(accelerators)

    # Try is necessary to ensure lock is released
    try:
        pymod_filename = f'{module_name}.py'
        # ...

        # Create new directories if not existing
        os.makedirs(folder, exist_ok=True)
        os.makedirs(epyccel_dirpath, exist_ok=True)

        # Store python file in '__epyccel__' folder, so that execute_pyccel can run
        with open(pymod_filename, 'w', encoding='utf-8') as f:
            f.writelines(code)

        # Generate shared library
        execute_pyccel(pymod_filename,
                       verbose         = verbose,
                       time_execution  = time_execution,
                       language        = language,
                       compiler_family = compiler_family_or_config,
                       flags           = flags,
                       wrapper_flags   = wrapper_flags,
                       include         = include,
                       libdir          = libdir,
                       modules         = (),
                       libs            = libs,
                       debug           = debug,
                       accelerators    = accelerators,
                       output_name     = module_name,
                       conda_warnings  = conda_warnings,
                       context_dict    = context_dict)


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
def epyccel(
    function_class_or_module,
    *,
    language        = 'fortran',
    compiler_family = None,
    compiler_config = None,
    flags           = None,
    wrapper_flags   = None,
    debug           = None,
    include         = (),
    libdir          = (),
    libs            = (),
    folder          = None,
    mpi             = False,
    openmp          = False,
#    openacc         = False,  # [YG, 17.06.2025] OpenACC is not supported yet
    verbose         = 0,
    time_execution  = False,
    developer_mode  = False,
    conda_warnings  = 'basic',
    context_dict    = None,
    comm            = None,
    root            = 0,
    bcast           = True,
    ):
    """
    Accelerate Python function or module using Pyccel in "embedded" mode.

    This function accelerates a Python function or module using Pyccel in "embedded" mode.
    It generates optimized code in the specified language (default is 'fortran')
    and compiles it for improved performance. Please be aware that only one of
    the parameters `compiler_family` and `compiler_config` may be provided.

    Parameters
    ----------
    function_class_or_module : function | class | module | str
        Python function, class, or module to be accelerated.
        If a string is passed then it is assumed to be the code from a module which
        should be accelerated. The module must be capable of running as a standalone
        file so it must include any necessary import statements.
    language : {'fortran', 'c', 'python'}
        Language of generated code (default: 'fortran').
    compiler_family : {'GNU', 'intel', 'PGI', 'nvidia', 'LLVM'}, optional
        Compiler family for which Pyccel uses a default configuration (default: 'GNU').
    compiler_config : pathlib.Path | str, optional
        Path to a JSON file containing a compiler configuration (overrides compiler_family).
    flags : iterable of str, optional
        Compiler flags.
    wrapper_flags : iterable of str, optional
        Compiler flags for the wrapper.
    debug : bool, optional
        Indicates whether the file should be compiled in debug mode.
        The default value is taken from the environment variable PYCCEL_DEBUG_MODE.
        If no such environment variable exists then the default is False.
    include : tuple, optional
        Additional include directories for the compiler.
    libdir : tuple, optional
        Additional library directories for the compiler.
    libs : tuple, optional
        Additional libraries to link with.
    folder : str, optional
        Output folder for the compiled code.
    mpi : bool, default=False
        If True, use MPI for parallel execution.
    openmp : bool, default=False
        If True, use OpenMP for parallel execution.
    verbose : int, default=0
        Set the level of verbosity to see additional information about the Pyccel process.
    time_execution : bool
        Time the execution of Pyccel's internal stages.
    developer_mode : bool, default=False
        If True, set error mode to developer.
    conda_warnings : {'off', 'basic', 'verbose'}
        Specify the level of Conda warnings to display (default: 'basic').
    context_dict : dict[str, obj], optional
        A dictionary containing any Python objects from the calling scope which should
        be made available to the translated code. By default any objects that are used
        in the body of the function are made available, as well as any global objects.
        If the argument is provided then these objects will be treated as additional
        to the default arguments.

    Returns
    -------
    object
        Accelerated function, class or module.

    Other Parameters
    ----------------
    comm : mpi4py.MPI.Comm, optional
        MPI communicator for calling Pyccel in parallel mode (default: None) (for parallel mode).
    root : int, optional
        MPI rank of process in charge of accelerating code (default: 0) (for parallel mode).
    bcast : {True, False}
        If False, only root process loads accelerated function/module (default: True) (for parallel mode). 

    See Also 
    -------- 
    epyccel_seq
        The version of this function called in a sequential context.

    Examples
    --------
    >>> def one(): return 1
    >>> from pyccel import epyccel
    >>> one_f = epyccel(one, language='fortran')
    >>> one_c = epyccel(one, language='c')
    """
    assert isinstance(function_class_or_module, (FunctionType, type, ModuleType, str))

    if None not in (compiler_family, compiler_config):
        raise TypeError('> Only one of the parameters `compiler_family` or `compiler_config` may be provided')

    err_mode = ErrorsMode()
    if developer_mode:
        err_mode.set_mode('developer')
    else:
        err_mode.set_mode(os.environ.get('PYCCEL_ERROR_MODE', 'user'))

    # Parallel version
    if comm is not None:

        from mpi4py import MPI
        from tblib  import pickling_support   # [YG, 27.10.2020] We use tblib to
        pickling_support.install()            # pickle tracebacks, which allows
                                              # mpi4py to broadcast exceptions
        assert isinstance(comm, MPI.Comm)
        assert isinstance(root, int)

        # Master process calls epyccel
        if comm.rank == root:
            try:
                mod, obj = epyccel_seq(
                    function_class_or_module,
                    language        = language,
                    compiler_family = compiler_family,
                    compiler_config = compiler_config,
                    flags           = flags,
                    wrapper_flags   = wrapper_flags,
                    include         = include,
                    libdir          = libdir,
                    libs            = libs,
                    folder          = folder,
                    mpi             = True,
                    openmp          = openmp,
                    openacc         = False,  # [YG, 17.06.2025] OpenACC is not supported yet
                    verbose         = verbose,
                    time_execution  = time_execution,
                    debug           = debug,
                    conda_warnings  = conda_warnings,
                    context_dict    = context_dict
                )
                mod_path = os.path.abspath(mod.__file__)
                mod_name = mod.__name__
                obj_name = function_class_or_module.__name__ if obj else None
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
            mod_path = comm.bcast(mod_path, root=root)
            mod_name = comm.bcast(mod_name, root=root)
            obj_name = comm.bcast(obj_name, root=root)

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
            mod, obj = epyccel_seq(
                    function_class_or_module,
                    language        = language,
                    compiler_family = compiler_family,
                    compiler_config = compiler_config,
                    flags           = flags,
                    wrapper_flags   = wrapper_flags,
                    include         = include,
                    libdir          = libdir,
                    libs            = libs,
                    folder          = folder,
                    mpi             = mpi,
                    openmp          = openmp,
                    openacc         = False,  # [YG, 17.06.2025] OpenACC is not supported yet
                    verbose         = verbose,
                    time_execution  = time_execution,
                    debug           = debug,
                    conda_warnings  = conda_warnings,
                    context_dict    = context_dict
                )
        except PyccelError as e:
            raise type(e)(str(e)) from None

    # Return accelerated function or class (if any), otherwise module
    return obj or mod
