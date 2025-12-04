# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
Contains the `execute_pyccel_make` function which carries out the main steps required to execute Pyccel on a multi-file project.
"""

import os
import shutil
import sys
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

__all__ = ['execute_pyccel_make']

build_system_handler = {'cmake': CMakeHandler,
                        'meson': MesonHandler}

#==============================================================================
# NOTE:
# [..]_dirname is the name of a directory
# [..]_dirpath is the full (absolute) path of a directory

def execute_pyccel_make(files, *,
                   verbose,
                   time_execution,
                   folder,
                   language,
                   compiler_family,
                   build_system,
                   debug = None,
                   accelerators,
                   conda_warnings,
                   build_code):
    """
    Run Pyccel-make on the provided files.

    Carry out the main steps required to execute Pyccel:
    - Parses the python file (syntactic stage)
    - Annotates the abstract syntax tree (semantic stage)
    - Generates the translated file(s) (codegen stage)
    - Generates the wrapper file(s) (wrapper stage)
    - Generates the build system file(s)
    - Compiles the files to generate executable(s) and/or shared library(s).

    Parameters
    ----------
    files : list[Path]
        The Python files to be translated.
    verbose : int
        Indicates the level of verbosity.
    time_execution : bool
        Show the time spent in each of Pyccel's internal stages.
    folder : str
        Path to the directory where output files are created. Generated files are created
        following the same folder structure as found in the folder where the command line
        tool is called. Default is the folder where the command line tool is called.
        The __pyccel__ directory is created inside this folder.
    language : str
        The target language Pyccel is translating to.
    compiler_family : str
        The compiler used to compile the generated files.
        This can also contain the name of a json file describing a compiler.
    build_system : str
        The build-system used to compile the generated files.
    debug : bool, optional
        Indicates whether the file should be compiled in debug mode.
        The default value is taken from the environment variable PYCCEL_DEBUG_MODE.
        If no such environment variable exists then the default is False.
    accelerators : iterable
        Tool used to accelerate the code (e.g., OpenMP, OpenACC).
    conda_warnings : str
        Specify the level of Conda warnings to display (choices: off, basic, verbose).
    build_code : bool
        Indicates if the build commands should be run.
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

    language = language.lower()

    Compiler.acceptable_bin_paths = get_condaless_search_path(conda_warnings)
    compiler = Compiler(compiler_family, debug)

    # Get compiler object
    Scope.name_clash_checker = name_clash_checkers[language]

    start_syntax = time.time()
    timers["Initialisation"] = start_syntax-start

    if len(files) == 0:
        errors.report("No files passed to pyccel make",
                      severity='fatal')

    cwd = os.getcwd()
    files = [f.relative_to(cwd) if f.is_absolute() else f for f in files]

    parsers = {f: Parser(f.absolute(), output_folder = folder) for f in files}

    to_remove = []
    for f, p in parsers.items():
        # Filter out empty __init__.py files
        if f.stem == '__init__' and len(p.fst.body) == 0:
            to_remove.append(f)
            continue
        # Parse Python file
        p.parse(verbose=verbose, d_parsers_by_filename = {f.absolute(): p for f, p in parsers.items()})
        if errors.has_errors():
            raise PyccelSyntaxError('Syntax step failed')

    for r in to_remove:
        parsers.pop(r)

    timers["Syntactic Stage"] = time.time() - start_syntax

    start_semantic = time.time()
    # Annotate abstract syntax Tree
    for f, p in parsers.items():
        p.annotate(verbose = verbose)

        if errors.has_errors():
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
        fname, prog_name = codegen.export(fname)

        if errors.has_errors():
            raise PyccelCodegenError('Code generation failed')

        timers["Codegen Stage"] += time.time() - start_codegen

        codegens.append(codegen)

        if language == 'python':
            new_location = folder / f
            if verbose:
                print(f"cp {fname} {new_location}")
            shutil.copyfile(fname, new_location)
        else:
            start_wrapper_creation = time.time()
            wrappergen = Wrappergen(codegen, codegen.name, language, verbose)
            wrappergen.wrap(str((base_dirpath / f).parent))
            timers['Wrapper creation'] += time.time() - start_wrapper_creation

            start_wrapper_printing = time.time()
            wrapper_files = wrappergen.print(output_dir)

            if errors.has_errors():
                raise PyccelCodegenError('Code generation step failed')

            timers['Wrapper printing'] += time.time() - start_wrapper_printing

            wrappergens.append(wrappergen)

            printer_imports.update(codegen.get_printer_imports())
            for i in wrappergen.get_additional_imports():
                printer_imports.update(i)
            printed_languages.update(wrappergen.printed_languages)

            relative_name = Path(fname).relative_to(pyccel_dirpath).with_suffix('')
            target_name = '__'.join(relative_name.parts) if has_conflicting_modules \
                          else relative_name.stem
            targets[f.absolute()] = CompileTarget(target_name,
                                                  f.absolute(), fname, dict(zip(wrapper_files, wrappergen.get_additional_imports())),
                                                  prog_name, codegen.get_printer_imports())

    if language == 'python':
        # Change working directory back to starting point
        pyccel_stage.pyccel_finished()
        if time_execution:
            print_timers(start, timers)
        return

    start_build_system_printing = time.time()

    for f, p in parsers.items():
        targets[f.absolute()].add_dependencies(*(targets[s.filename] for s in p.sons))

    stdlib_deps = {}
    for t in targets.values():
        manage_dependencies(printer_imports, pyccel_dirpath = pyccel_dirpath, compiler = compiler,
                            language = language, verbose = verbose, convert_only = True, mod_obj = t,
                            installed_libs = stdlib_deps)

    build_project = BuildProject(base_dirpath, targets.values(), printed_languages,
                                 stdlib_deps)

    build_sys = build_system_handler[build_system](pyccel_dirpath, base_dirpath, folder,
                                                   verbose = verbose, debug_mode = debug,
                                                   compiler = compiler, accelerators = accelerators)

    build_sys.generate(build_project)

    timers['Build system printing'] = time.time() - start_build_system_printing

    if build_code:
        start_compilation = time.time()
        build_sys.compile()
        timers['Compilation'] = time.time() - start_compilation

    # Change working directory back to starting point
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
