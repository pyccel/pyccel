# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
Contains the execute_pyccel function which carries out the main steps required to execute pyccel
"""

import os
import sys
import shutil
import time
from pathlib import Path

from pyccel.errors.errors          import Errors, PyccelError
from pyccel.errors.errors          import PyccelSyntaxError, PyccelSemanticError, PyccelCodegenError
from pyccel.errors.messages        import PYCCEL_RESTRICTION_TODO
from pyccel.parser.parser          import Parser
from pyccel.codegen.codegen        import Codegen
from pyccel.codegen.utilities      import manage_dependencies, get_module_and_compile_dependencies
from pyccel.codegen.wrappergen     import Wrappergen
from pyccel.naming                 import name_clash_checkers
from pyccel.utilities.stage        import PyccelStage
from pyccel.ast.utilities          import python_builtin_libs
from pyccel.parser.scope           import Scope

from .compiling.basic     import CompileObj
from .compiling.compilers import Compiler, get_condaless_search_path

pyccel_stage = PyccelStage()

__all__ = ['execute_pyccel']

#==============================================================================
# NOTE:
# [..]_dirname is the name of a directory
# [..]_dirpath is the full (absolute) path of a directory

# TODO: change name of variable 'module_name', as it could be a program
# TODO [YG, 04.02.2020]: check if we should catch BaseException instead of Exception
def execute_pyccel_wrap(fname, *,
                   syntax_only     = False,
                   semantic_only   = False,
                   convert_only    = False,
                   verbose         = 0,
                   time_execution  = False,
                   folder          = None,
                   language        = None,
                   compiler_family = None,
                   flags           = None,
                   wrapper_flags   = None,
                   include         = (),
                   libdir          = (),
                   modules         = (),
                   libs            = (),
                   debug           = None,
                   accelerators    = (),
                   output_name     = None,
                   compiler_export_file = None,
                   conda_warnings  = 'basic',
                   context_dict    = None):
    """
    Run Pyccel on the provided code.

    Carry out the main steps required to execute Pyccel:
    - Parses the python file (syntactic stage)
    - Annotates the abstract syntax tree (semantic stage)
    - Generates the translated file(s) (codegen stage)
    - Compiles the files to generate an executable and/or a shared library.

    Parameters
    ----------
    fname : str
        Name of the Python file to be translated.
    syntax_only : bool, optional
        Indicates whether the pipeline should stop after the syntax stage. Default is False.
    semantic_only : bool, optional
        Indicates whether the pipeline should stop after the semantic stage. Default is False.
    convert_only : bool, optional
        Indicates whether the pipeline should stop after the codegen stage. Default is False.
    verbose : int, default=0
        Indicates the level of verbosity.
    time_execution : bool, default=False
        Show the time spent in each of Pyccel's internal stages.
    folder : str, optional
        Path to the working directory. Default is the folder containing the file to be translated.
    language : str, optional
        The target language Pyccel is translating to. Default is 'fortran'.
    compiler_family : str, optional
        The compiler used to compile the generated files. Default is 'GNU'.
        This can also contain the name of a json file describing a compiler.
    flags : str, optional
        The flags passed to the compiler. Default is provided by the Compiler.
    wrapper_flags : str, optional
        The flags passed to the compiler to compile the C wrapper. Default is provided by the Compiler.
    include : list, optional
        List of include directory paths.
    libdir : list, optional
        List of paths to directories containing the required libraries.
    modules : list, optional
        List of files that must be compiled in order to compile this module.
    libs : list, optional
        List of required libraries.
    debug : bool, optional
        Indicates whether the file should be compiled in debug mode.
        The default value is taken from the environment variable PYCCEL_DEBUG_MODE.
        If no such environment variable exists then the default is False.
    accelerators : iterable, optional
        Tool used to accelerate the code (e.g., OpenMP, OpenACC).
    output_name : str, optional
        Name of the generated module. Default is the same name as the translated file.
    compiler_export_file : str, optional
        Name of the JSON file to which compiler information is exported. Default is None.
    conda_warnings : str, optional
        Specify the level of Conda warnings to display (choices: off, basic, verbose), Default is 'basic'.
    context_dict : dict[str, object], optional
        A dictionary containing any variables that are available in the calling context.
        This can allow certain constants to be defined outside of the function passed to epyccel.
    """
    start = time.time()
    timers = {}
    if fname.endswith('.pyh'):
        syntax_only = True
        if verbose:
            print("Header file recognised, stopping after syntactic stage")

    if Path(fname).stem in python_builtin_libs:
        raise ValueError(f"File called {os.path.basename(fname)} has the same name as a Python built-in package and can't be imported from Python. See #1402")

    # Reset Errors singleton before parsing a new file
    errors = Errors()
    errors.reset()

    # TODO [YG, 03.02.2020]: test validity of function arguments

    # Copy list arguments to local lists to avoid unexpected behavior
    include = [os.path.abspath(i) for i in include]
    libdir  = [os.path.abspath(l) for l in libdir]
    modules  = [*modules]
    libs     = [*libs]


    # Store current directory and add it to sys.path
    # variable to imitate Python's import behavior
    base_dirpath = os.getcwd()
    sys.path.insert(0, base_dirpath)

    # Unified way to handle errors: print formatted error message, then move
    # to original working directory. Caller should then raise exception.
    def handle_error(stage):
        print('\nERROR at {} stage'.format(stage))
        errors.check()
        os.chdir(base_dirpath)

    # Identify absolute path, directory, and filename
    pymod_filepath = os.path.abspath(fname)
    pymod_dirpath, pymod_filename = os.path.split(pymod_filepath)
    if compiler_export_file:
        compiler_export_file = os.path.abspath(compiler_export_file)

    # Extract module name
    module_name = os.path.splitext(pymod_filename)[0]

    # Define working directory 'folder'
    if folder is None or folder == "":
        folder = pymod_dirpath
    else:
        folder = os.path.abspath(folder)

    # Define default debug mode
    if debug is None:
        debug = bool(os.environ.get('PYCCEL_DEBUG_MODE', False))

    # Define directory name and path for pyccel & cpython build
    pyccel_dirname = '__pyccel__' + os.environ.get('PYTEST_XDIST_WORKER', '')
    pyccel_dirpath = os.path.join(folder, pyccel_dirname)

    # Create new directories if not existing
    os.makedirs(folder, exist_ok=True)
    if not (syntax_only or semantic_only):
        os.makedirs(pyccel_dirpath, exist_ok=True)

    if conda_warnings not in ('off', 'basic', 'verbose'):
        raise ValueError("conda warnings accept {off, basic,verbose}")

    if language is None:
        language = 'fortran'
    else:
        language = language.lower()

    # Choose Fortran compiler
    if compiler_family is None:
        compiler_family = os.environ.get('PYCCEL_DEFAULT_COMPILER', 'GNU')

    flags = [] if flags is None else flags.split()
    wrapper_flags = [] if wrapper_flags is None else wrapper_flags.split()

    # Get compiler object
    Compiler.acceptable_bin_paths = get_condaless_search_path(conda_warnings)
    compiler = Compiler(compiler_family, debug)

    # Export the compiler information if requested
    if compiler_export_file:
        compiler.export_compiler_info(compiler_export_file)
        if not fname:
            return

    Scope.name_clash_checker = name_clash_checkers[language]

    # Change working directory to 'folder'
    os.chdir(folder)

    start_syntax = time.time()
    timers["Initialisation"] = start_syntax-start
    # Parse Python file
    try:
        parser = Parser(pymod_filepath, output_folder = folder, context_dict = context_dict)
        parser.parse(verbose=verbose)
    except NotImplementedError as error:
        msg = str(error)
        errors.report(msg+'\n'+PYCCEL_RESTRICTION_TODO,
            severity='error',
            traceback=error.__traceback__)
    except PyccelError:
        handle_error('parsing (syntax)')
        raise
    if errors.has_errors():
        handle_error('parsing (syntax)')
        raise PyccelSyntaxError('Syntax step failed')

    timers["Syntactic Stage"] = time.time() - start_syntax

    if syntax_only:
        pyccel_stage.pyccel_finished()
        if time_execution:
            print_timers(start, timers)
        return

    start_semantic = time.time()
    # Annotate abstract syntax Tree
    try:
        parser.annotate(verbose = verbose)
    except NotImplementedError as error:
        msg = str(error)
        errors.report(msg+'\n'+PYCCEL_RESTRICTION_TODO,
            severity='error',
            traceback=error.__traceback__)
    except PyccelError:
        handle_error('annotation (semantic)')
        # Raise a new error to avoid a large traceback
        raise PyccelSemanticError('Semantic step failed') from None

    if errors.has_errors():
        handle_error('annotation (semantic)')
        raise PyccelSemanticError('Semantic step failed')

    timers["Semantic Stage"] = time.time() - start_semantic

    if semantic_only:
        pyccel_stage.pyccel_finished()
        if time_execution:
            print_timers(start, timers)
        return

    # -------------------------------------------------------------------------

    semantic_parser = parser.semantic_parser

    codegen = Codegen(semantic_parser, module_name, language, verbose)

    start_wrapper_creation = time.time()
    wrappergen = Wrappergen(codegen, codegen.name, language, verbose)
    wrappergen.wrap(base_dirpath)
    timers['Wrapper creation'] = time.time() - start_wrapper_creation

    start_wrapper_printing = time.time()
    try:
        wrapper_files = wrappergen.print(pyccel_dirpath)
    except NotImplementedError as error:
        errors.report(f'{error}\n'+PYCCEL_RESTRICTION_TODO,
            severity='error',
            traceback=error.__traceback__)
        handle_error('code generation (wrapping)')
        raise PyccelCodegenError(msg) from None
    except PyccelError:
        handle_error('code generation (wrapping)')
        raise
    timers['Wrapper printing'] = time.time() - start_wrapper_printing

    printed_languages = wrappergen.printed_languages

    includes = [os.path.join(pymod_dirpath, i.strip()) for i in parser.metavars.get('includes', '').split(',') if i]
    libdirs = [os.path.join(pymod_dirpath, l.strip()) for l in parser.metavars.get('libdirs', '').split(',') if l]
    libs = [l.strip() for l in parser.metavars.get('libraries', '').split(',') if l]

    meta_flags = [f.strip() for f in parser.metavars.get('flags', '').split(',') if f]

    wrapper_compile_objs = [CompileObj(filepath,
                                       pyccel_dirpath,
                                       flags = meta_flags + flags,
                                       include = includes,
                                       libs = libs,
                                       libdir = libdirs)
                            for filepath in wrapper_files[:-1]] + \
                           [CompileObj(wrapper_files[-1],
                                       pyccel_dirpath,
                                       flags = meta_flags + wrapper_flags,
                                       include = includes,
                                       libs = libs,
                                       libdir = libdirs,
                                       extra_compilation_tools = ('python',))]

    for i, (obj, lang, imports) in enumerate(zip(wrapper_compile_objs, printed_languages, wrappergen.get_additional_imports())):
        obj.add_dependencies(*wrapper_compile_objs[:i])
        manage_dependencies(imports, compiler,
                pyccel_dirpath, obj, lang, verbose)

    start_compile_wrapper = time.time()
    for obj, wrapper_language in zip(wrapper_compile_objs, printed_languages):
        compiler.compile_module(compile_obj=obj,
                output_folder=pyccel_dirpath,
                language=wrapper_language,
                verbose=verbose)

    if output_name is None:
        output_name = str(Path(fname).with_suffix(''))

    sharedlib_filepath = compiler.compile_shared_library(wrapper_compile_objs[-1],
                                                    output_folder = folder,
                                                    sharedlib_modname = output_name,
                                                    language = language,
                                                    verbose = verbose)

    timers['Wrapper compilation'] = time.time() - start_compile_wrapper


    # Print all warnings now
    if errors.has_warnings():
        errors.check()

    # Change working directory back to starting point
    os.chdir(base_dirpath)
    pyccel_stage.pyccel_finished()

