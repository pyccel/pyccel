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
from pyccel.codegen.build_generation.cmake_gen import CMakeHandler
from pyccel.codegen.build_generation.meson_gen import MesonHandler
from pyccel.codegen.codegen        import Codegen
from pyccel.codegen.compiling.project import CompileTarget, BuildProject
from pyccel.codegen.utilities      import manage_dependencies
from pyccel.codegen.wrappergen     import Wrappergen
from pyccel.naming                 import name_clash_checkers
from pyccel.utilities.stage        import PyccelStage
from pyccel.parser.scope           import Scope

from .compiling.compilers import Compiler, get_condaless_search_path

pyccel_stage = PyccelStage()

__all__ = ['execute_pyccel']

build_system_handler = {'cmake': CMakeHandler,
                        'meson': MesonHandler}

#==============================================================================
# NOTE:
# [..]_dirname is the name of a directory
# [..]_dirpath is the full (absolute) path of a directory

# TODO: change name of variable 'module_name', as it could be a program
# TODO [YG, 04.02.2020]: check if we should catch BaseException instead of Exception
def execute_pyccel_make(files, *,
                   verbose         = 0,
                   time_execution  = False,
                   folder          = None,
                   language        = None,
                   compiler_family = None,
                   build_system    = None,
                   debug           = None,
                   accelerators    = (),
                   conda_warnings  = 'basic'):
    """
    Run Pyccel on the provided code.

    Carry out the main steps required to execute Pyccel:
    - Parses the python file (syntactic stage)
    - Annotates the abstract syntax tree (semantic stage)
    - Generates the translated file(s) (codegen stage)
    - Compiles the files to generate an executable and/or a shared library.

    Parameters
    ----------
    files : list[Path]
        The Python files to be translated.
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
    debug : bool, optional
        Indicates whether the file should be compiled in debug mode.
        The default value is taken from the environment variable PYCCEL_DEBUG_MODE.
        If no such environment variable exists then the default is False.
    accelerators : iterable, optional
        Tool used to accelerate the code (e.g., OpenMP, OpenACC).
    conda_warnings : str, optional
        Specify the level of Conda warnings to display (choices: off, basic, verbose), Default is 'basic'.
    context_dict : dict[str, object], optional
        A dictionary containing any variables that are available in the calling context.
        This can allow certain constants to be defined outside of the function passed to epyccel.
    """
    start = time.time()
    timers = {}

    # Reset Errors singleton before parsing a new file
    errors = Errors()
    errors.reset()

    # Store current directory and add it to sys.path
    # variable to imitate Python's import behavior
    base_dirpath = os.getcwd()
    sys.path.insert(0, base_dirpath)

    # Unified way to handle errors: print formatted error message, then move
    # to original working directory. Caller should then raise exception.
    def handle_error(stage):
        print(f'\nERROR at {stage} stage')
        errors.check()
        os.chdir(base_dirpath)

    # Define working directory 'folder'
    if folder is None or folder == "":
        folder = Path(os.getcwd())
    else:
        folder = folder.absolute()

    # Define default debug mode
    if debug is None:
        debug = bool(os.environ.get('PYCCEL_DEBUG_MODE', False))

    # Define directory name and path for pyccel & cpython build
    pyccel_dirname = '__pyccel__' + os.environ.get('PYTEST_XDIST_WORKER', '')
    pyccel_dirpath = folder / pyccel_dirname

    # Create new directories if not existing
    os.makedirs(folder, exist_ok=True)
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

    Compiler.acceptable_bin_paths = get_condaless_search_path(conda_warnings)
    compiler = Compiler(compiler_family, debug)

    # Get compiler object
    Scope.name_clash_checker = name_clash_checkers[language]

    # Change working directory to 'folder'
    os.chdir(folder)

    start_syntax = time.time()
    timers["Initialisation"] = start_syntax-start

    parsers = {f: Parser(f.absolute(), output_folder = folder) for f in files}

    to_remove = []
    for f, p in parsers.items():
        # Filter out empty __init__.py files
        if f.stem == '__init__' and len(p.fst.body) == 0:
            to_remove.append(f)
            continue
        # Parse Python file
        try:
            p.parse(verbose=verbose, d_parsers_by_filename = {f.absolute(): p for f, p in parsers.items()})
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

    for r in to_remove:
        parsers.pop(r)

    timers["Syntactic Stage"] = time.time() - start_syntax

    start_semantic = time.time()
    # Annotate abstract syntax Tree
    for f, p in parsers.items():
        try:
            p.annotate(verbose = verbose)
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

    # -------------------------------------------------------------------------

    timers["Codegen Stage"] = 0
    timers['Wrapper creation'] = 0
    timers['Wrapper printing'] = 0

    codegens = []
    wrappergens = []
    printer_imports = {'cwrapper': None}
    targets = {}
    printed_languages = set()
    has_conflicting_modules = len({p.stem for p in parsers}) != len(parsers)
    for f, p in parsers.items():
        semantic_parser = p.semantic_parser
        start_codegen = time.time()
        # Generate low-level code file
        codegen = Codegen(semantic_parser, f, language, verbose)
        fname = (pyccel_dirpath / f).with_suffix('')
        output_dir = fname.parent
        os.makedirs(output_dir, exist_ok=True)
        try:
            fname, prog_name = codegen.export(fname)
        except NotImplementedError as error:
            msg = str(error)
            errors.report(msg+'\n'+PYCCEL_RESTRICTION_TODO,
                severity='error',
                traceback=error.__traceback__)
        except PyccelError:
            handle_error('code generation')
            # Raise a new error to avoid a large traceback
            raise PyccelCodegenError('Code generation failed') from None

        if errors.has_errors():
            handle_error('code generation')
            raise PyccelCodegenError('Code generation failed')

        timers["Codegen Stage"] += time.time() - start_codegen

        codegens.append(codegen)

        if language != 'python':
            start_wrapper_creation = time.time()
            wrappergen = Wrappergen(codegen, codegen.name, language, verbose)
            wrappergen.wrap(base_dirpath)
            timers['Wrapper creation'] += time.time() - start_wrapper_creation

            start_wrapper_printing = time.time()
            wrapper_files = wrappergen.print(output_dir)
            timers['Wrapper printing'] += time.time() - start_wrapper_printing

            wrappergens.append(wrappergen)
            printer_imports.update(codegen.get_printer_imports())
            printed_languages.update(wrappergen.printed_languages)

            relative_name = Path(fname).relative_to(pyccel_dirpath).with_suffix('')
            target_name = '__'.join(relative_name.parts) if has_conflicting_modules \
                          else relative_name.stem
            targets[f.absolute()] = CompileTarget('__'.join(relative_name.parts),
                                                  f.absolute(), fname, wrapper_files,
                                                  prog_name, codegen.get_printer_imports())

    if language == 'python':
        # Change working directory back to starting point
        os.chdir(base_dirpath)
        pyccel_stage.pyccel_finished()
        if time_execution:
            print_timers(start, timers)
        return

    for f, p in parsers.items():
        targets[f.absolute()].add_dependencies(targets[s.filename] for s in p.sons)

    stdlib_deps = {}
    #TODO: Fix
    manage_dependencies(printer_imports, pyccel_dirpath = pyccel_dirpath, compiler = compiler,
                        language = language, verbose = verbose, convert_only = True,
                        installed_libs = stdlib_deps)

    try:
        build_project = BuildProject(base_dirpath, targets.values(), printed_languages,
                                     stdlib_deps)

        build_sys = build_system_handler[build_system](pyccel_dirpath, base_dirpath, verbose, debug)

        build_sys.generate(build_project)
    except NotImplementedError as error:
        msg = str(error)
        errors.report(msg+'\n'+PYCCEL_RESTRICTION_TODO,
            severity='error',
            traceback=error.__traceback__)
    except PyccelError:
        handle_error('code generation')
        # Raise a new error to avoid a large traceback
        raise PyccelCodegenError('Code generation failed') from None

    if errors.has_errors():
        handle_error('build system generation')
        raise PyccelCodegenError('Build system generation failed')

    build_sys.compile()

    # Print all warnings now
    if errors.has_warnings():
        errors.check()

    # Change working directory back to starting point
    os.chdir(base_dirpath)
    pyccel_stage.pyccel_finished()

    if time_execution:
        print_timers(start, timers)

def print_timers(start, timers):
    """
    Print the timers measured during the execution.

    Print the timers measured during the execution.

    Parameters
    ----------
    start : float
        The start time for the execution.
    timers : dict
        A dictionary containing the times measured.
    """
    timers['Total'] = time.time()-start
    print("-------------------- Timers -------------------------")
    for n,t in timers.items():
        print(f'{n:<30}: ',t)
