# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
Contains the `execute_pyccel_wrap` function which carries out the main steps required
to execute the `pyccel wrap` command.
"""

import os
import sys
import time

from pyccel.errors.errors          import Errors, PyccelError
from pyccel.errors.errors          import PyccelSyntaxError, PyccelSemanticError, PyccelCodegenError
from pyccel.parser.parser          import Parser
from pyccel.codegen.codegen        import Codegen
from pyccel.codegen.pipeline       import print_timers
from pyccel.codegen.utilities      import manage_dependencies
from pyccel.codegen.wrappergen     import Wrappergen
from pyccel.naming                 import name_clash_checkers
from pyccel.utilities.stage        import PyccelStage
from pyccel.ast.utilities          import python_builtin_libs
from pyccel.parser.scope           import Scope

from .compiling.basic     import CompileObj
from .compiling.compilers import Compiler, get_condaless_search_path

pyccel_stage = PyccelStage()

__all__ = ['execute_pyccel_wrap']

#==============================================================================
# NOTE:
# [..]_dirname is the name of a directory
# [..]_dirpath is the full (absolute) path of a directory

# TODO: change name of variable 'module_name', as it could be a program
# TODO [YG, 04.02.2020]: check if we should catch BaseException instead of Exception
def execute_pyccel_wrap(fname, *,
                   convert_only,
                   verbose,
                   time_execution,
                   folder,
                   language,
                   compiler_family,
                   debug,
                   accelerators,
                   conda_warnings):
    """
    Run Pyccel on the provided code.

    Carry out the main steps required to execute Pyccel:
    - Parses the python file (syntactic stage)
    - Annotates the abstract syntax tree (semantic stage)
    - Generates the translated file(s) (codegen stage)
    - Compiles the files to generate an executable and/or a shared library.

    Parameters
    ----------
    fname : Path
        Name of the stub file describing the mapping between Python and low-level code.
    convert_only : bool
        Indicates whether the pipeline should stop after generating the wrapper files.
    verbose : int
        Indicates the level of verbosity.
    time_execution : bool
        Show the time spent in each of Pyccel's internal stages.
    folder : Path
        Path to the working directory. Default is the folder containing the file to be translated.
    language : str
        The target language Pyccel is translating to.
    compiler_family : str
        The compiler used to compile the generated files.
        This can also contain the name of a json file describing a compiler.
    debug : bool
        Indicates whether the wrapper files should be compiled in debug mode.
        The default value is taken from the environment variable PYCCEL_DEBUG_MODE.
        If no such environment variable exists then the default is False.
    accelerators : iterable
        Tool used to accelerate the code (e.g., OpenMP, OpenACC).
    conda_warnings : str
        Specify the level of Conda warnings to display (choices: off, basic, verbose), Default is 'basic'.
    """
    start = time.time()
    timers = {}
    if fname.stem in python_builtin_libs:
        raise ValueError(f"File called {fname.name} has the same name as a Python built-in package and can't be imported from Python. See #1402")

    # Reset Errors singleton before parsing a new file
    errors = Errors()
    errors.reset()

    output_name = fname.stem

    # Store current directory and add it to sys.path
    # variable to imitate Python's import behavior
    base_dirpath = os.getcwd()
    sys.path.insert(0, base_dirpath)

    # Identify absolute path, directory, and filename
    pymod_filepath = fname.absolute()
    pymod_dirpath = fname.parent

    # Extract module name
    module_name = fname.stem

    # Define working directory 'folder'
    folder = folder.absolute() if folder else pymod_dirpath

    # Define default debug mode
    if debug is None:
        debug = bool(os.environ.get('PYCCEL_DEBUG_MODE', False))

    # Define directory name and path for pyccel & cpython build
    pyccel_dirname = '__pyccel__' + os.environ.get('PYTEST_XDIST_WORKER', '')
    pyccel_dirpath = folder / pyccel_dirname

    # Create new directories if not existing
    os.makedirs(pyccel_dirpath, exist_ok=True)
    os.makedirs(folder, exist_ok=True)

    if conda_warnings not in ('off', 'basic', 'verbose'):
        raise ValueError("conda warnings accept {off, basic,verbose}")

    language = language.lower()

    # Get compiler object
    Compiler.acceptable_bin_paths = get_condaless_search_path(conda_warnings)
    compiler = Compiler(compiler_family, debug)

    Scope.name_clash_checker = name_clash_checkers[language]

    start_syntax = time.time()
    timers["Initialisation"] = start_syntax-start
    # Parse Python file
    parser = Parser(pymod_filepath, output_folder = folder, context_dict = {})
    parser.parse(verbose=verbose)

    if errors.has_errors():
        raise PyccelSyntaxError('Syntax step failed')

    timers["Syntactic Stage"] = time.time() - start_syntax

    start_semantic = time.time()
    # Annotate abstract syntax Tree
    parser.annotate(verbose = verbose)

    if errors.has_errors():
        raise PyccelSemanticError('Semantic step failed')

    timers["Semantic Stage"] = time.time() - start_semantic

    # -------------------------------------------------------------------------

    semantic_parser = parser.semantic_parser

    codegen = Codegen(semantic_parser, module_name, language, verbose)

    start_wrapper_creation = time.time()
    wrappergen = Wrappergen(codegen, codegen.name, language, verbose)
    wrappergen.wrap(str(folder))
    timers['Wrapper creation'] = time.time() - start_wrapper_creation

    start_wrapper_printing = time.time()
    wrapper_files = wrappergen.print(pyccel_dirpath)

    if errors.has_errors():
        raise PyccelCodegenError('Code generation step failed')

    timers['Wrapper printing'] = time.time() - start_wrapper_printing

    if convert_only:
        pyccel_stage.pyccel_finished()
        if time_execution:
            print_timers(start, timers)
        return

    printed_languages = wrappergen.printed_languages

    includes = [os.path.join(pymod_dirpath, i.strip()) for i in parser.metavars.get('includes', '').split(',') if i]
    libdirs = [os.path.join(pymod_dirpath, l.strip()) for l in parser.metavars.get('libdirs', '').split(',') if l]
    libs = [l.strip() for l in parser.metavars.get('libraries', '').split(',') if l]

    flags = [f.strip() for f in parser.metavars.get('flags', '').split(',') if f]

    wrapper_compile_objs = [CompileObj(filepath,
                                       pyccel_dirpath,
                                       flags = flags,
                                       include = includes,
                                       libs = libs,
                                       libdir = libdirs)
                            for filepath in wrapper_files[:-1]] + \
                           [CompileObj(wrapper_files[-1],
                                       pyccel_dirpath,
                                       flags = flags,
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

    compiler.compile_shared_library(wrapper_compile_objs[-1],
                                                    output_folder = folder,
                                                    sharedlib_modname = output_name,
                                                    language = language,
                                                    verbose = verbose)

    timers['Wrapper compilation'] = time.time() - start_compile_wrapper

    # Reset pyccel stage
    pyccel_stage.pyccel_finished()

    if time_execution:
        print_timers(start, timers)

