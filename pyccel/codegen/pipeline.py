# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
Contains the execute_pyccel function which carries out the main steps required to execute pyccel
"""

import os
import sys
import shutil

from pyccel.errors.errors          import Errors, PyccelError
from pyccel.errors.errors          import PyccelSyntaxError, PyccelSemanticError, PyccelCodegenError
from pyccel.errors.messages        import PYCCEL_RESTRICTION_TODO
from pyccel.parser.parser          import Parser
from pyccel.codegen.codegen        import Codegen
from pyccel.codegen.utilities      import recompile_object
from pyccel.codegen.utilities      import copy_internal_library
from pyccel.codegen.utilities      import internal_libs
from pyccel.codegen.python_wrapper import create_shared_library
from pyccel.naming                 import name_clash_checkers
from pyccel.utilities.stage        import PyccelStage
from pyccel.parser.scope           import Scope

from .compiling.basic     import CompileObj
from .compiling.compilers import Compiler

pyccel_stage = PyccelStage()

__all__ = ['execute_pyccel']

#==============================================================================
# NOTE:
# [..]_dirname is the name of a directory
# [..]_dirpath is the full (absolute) path of a directory

# TODO: change name of variable 'module_name', as it could be a program
# TODO [YG, 04.02.2020]: check if we should catch BaseException instead of Exception
def execute_pyccel(fname, *,
                   syntax_only   = False,
                   semantic_only = False,
                   convert_only  = False,
                   verbose       = False,
                   folder        = None,
                   language      = None,
                   compiler      = None,
                   fflags        = None,
                   wrapper_flags = None,
                   includes      = (),
                   libdirs       = (),
                   modules       = (),
                   libs          = (),
                   debug         = False,
                   accelerators  = (),
                   output_name   = None,
                   compiler_export_file = None):
    """
    Pyccel internal.

    Carrie out the main steps required to execute pyccel
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
    verbose : bool, optional
        Indicates whether debugging messages should be printed. Default is False.
    folder : str, optional
        Path to the working directory. Default is the folder containing the file to be translated.
    language : str, optional
        The target language Pyccel is translating to. Default is 'fortran'.
    compiler : str, optional
        The compiler used to compile the generated files. Default is 'GNU'.
    fflags : str, optional
        The flags passed to the compiler. Default is provided by the Compiler.
    wrapper_flags : str, optional
        The flags passed to the compiler to compile the C wrapper. Default is provided by the Compiler.
    includes : list, optional
        List of include directory paths.
    libdirs : list, optional
        List of paths to directories containing the required libraries.
    modules : list, optional
        List of files that must be compiled in order to compile this module.
    libs : list, optional
        List of required libraries.
    debug : bool, optional
        Indicates whether the file should be compiled in debug mode. Default is False.
        (Currently, this only implies that the flag -fcheck=bounds is added.).
    accelerators : iterable, optional
        Tool used to accelerate the code (e.g., OpenMP, OpenACC).
    output_name : str, optional
        Name of the generated module. Default is the same name as the translated file.
    compiler_export_file : str, optional
        Name of the JSON file to which compiler information is exported. Default is None.
    """
    if fname.endswith('.pyh'):
        syntax_only = True
        if verbose:
            print("Header file recognised, stopping after syntactic stage")
    if fname == "test.py":
        raise ValueError("files called test can cause problems for some compilers and can't be imported from Python. See #1402")
    # Reset Errors singleton before parsing a new file
    errors = Errors()
    errors.reset()

    # TODO [YG, 03.02.2020]: test validity of function arguments

    # Copy list arguments to local lists to avoid unexpected behavior
    includes = [os.path.abspath(i) for i in includes]
    libdirs  = [os.path.abspath(l) for l in libdirs]
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

    # Define directory name and path for pyccel & cpython build
    pyccel_dirname = '__pyccel__'
    pyccel_dirpath = os.path.join(folder, pyccel_dirname)

    # Create new directories if not existing
    os.makedirs(folder, exist_ok=True)
    if not (syntax_only or semantic_only):
        os.makedirs(pyccel_dirpath, exist_ok=True)

    # Change working directory to 'folder'
    os.chdir(folder)

    if language is None:
        language = 'fortran'

    # Choose Fortran compiler
    if compiler is None:
        compiler = 'GNU'

    fflags = [] if fflags is None else fflags.split()
    wrapper_flags = [] if wrapper_flags is None else wrapper_flags.split()

    # Get compiler object
    src_compiler = Compiler(compiler, language, debug)
    wrapper_compiler = Compiler('GNU', 'c', debug)

    # Export the compiler information if requested
    if compiler_export_file:
        src_compiler.export_compiler_info(compiler_export_file)
        if not fname:
            return

    Scope.name_clash_checker = name_clash_checkers[language]

    # Parse Python file
    try:
        parser = Parser(pymod_filepath)
        parser.parse(verbose=verbose)
    except NotImplementedError as error:
        msg = str(error)
        errors.report(msg+'\n'+PYCCEL_RESTRICTION_TODO,
            severity='error')
    except PyccelError:
        handle_error('parsing (syntax)')
        raise
    if errors.has_errors():
        handle_error('parsing (syntax)')
        raise PyccelSyntaxError('Syntax step failed')

    if syntax_only:
        pyccel_stage.pyccel_finished()
        return

    # Annotate abstract syntax Tree
    try:
        settings = {'verbose':verbose}
        parser.annotate(**settings)
    except NotImplementedError as error:
        msg = str(error)
        errors.report(msg+'\n'+PYCCEL_RESTRICTION_TODO,
            severity='error')
    except PyccelError:
        handle_error('annotation (semantic)')
        # Raise a new error to avoid a large traceback
        raise PyccelSemanticError('Semantic step failed') from None

    if errors.has_errors():
        handle_error('annotation (semantic)')
        raise PyccelSemanticError('Semantic step failed')

    if semantic_only:
        pyccel_stage.pyccel_finished()
        return

    # -------------------------------------------------------------------------

    semantic_parser = parser.semantic_parser
    # Generate .f90 file
    try:
        codegen = Codegen(semantic_parser, module_name)
        fname = os.path.join(pyccel_dirpath, module_name)
        fname, prog_name = codegen.export(fname, language=language)
    except NotImplementedError as error:
        msg = str(error)
        errors.report(msg+'\n'+PYCCEL_RESTRICTION_TODO,
            severity='error')
    except PyccelError:
        handle_error('code generation')
        # Raise a new error to avoid a large traceback
        raise PyccelCodegenError('Code generation failed') from None

    if errors.has_errors():
        handle_error('code generation')
        raise PyccelCodegenError('Code generation failed')

    if language == 'python':
        output_file = (output_name + '.py') if output_name else os.path.basename(fname)
        new_location = os.path.join(folder, output_file)
        if verbose:
            print("cp {} {}".format(fname, new_location))
        shutil.copyfile(fname, new_location)

        # Change working directory back to starting point
        os.chdir(base_dirpath)
        pyccel_stage.pyccel_finished()
        return

    compile_libs = [*libs, parser.metavars['libraries']] \
                    if 'libraries' in parser.metavars else libs
    mod_obj = CompileObj(file_name = fname,
            folder       = pyccel_dirpath,
            flags        = fflags,
            includes     = includes,
            libs         = compile_libs,
            libdirs      = libdirs,
            dependencies = modules,
            accelerators = accelerators)
    parser.compile_obj = mod_obj

    #------------------------------------------------------
    # TODO: collect dependencies and proceed recursively
    # if recursive:
    #     for dep in parser.sons:
    #         # Call same function on 'dep'
    #         pass
    #------------------------------------------------------

    # Iterate over the internal_libs list and determine if the printer
    # requires an internal lib to be included.
    for lib_name, (stdlib_folder, stdlib) in internal_libs.items():
        if lib_name in codegen.get_printer_imports():

            lib_dest_path = copy_internal_library(stdlib_folder, pyccel_dirpath)

            # stop after copying lib to __pyccel__ directory for
            # convert only
            if convert_only:
                continue

            # Pylint determines wrong type
            stdlib.reset_folder(lib_dest_path) # pylint: disable=E1101
            # get the include folder path and library files
            recompile_object(stdlib,
                              compiler = src_compiler,
                              verbose  = verbose)

            mod_obj.add_dependencies(stdlib)

    if convert_only:
        # Change working directory back to starting point
        os.chdir(base_dirpath)
        pyccel_stage.pyccel_finished()
        return

    deps = dict()
    # ...
    # Determine all .o files and all folders needed by executable
    def get_module_dependencies(parser, deps):
        mod_folder = os.path.join(os.path.dirname(parser.filename), "__pyccel__")
        mod_base = os.path.basename(parser.filename)

        # Stop conditions
        if parser.metavars.get('module_name', None) == 'omp_lib':
            return

        if parser.compile_obj:
            deps[mod_base] = parser.compile_obj
        elif mod_base not in deps:
            compile_libs = (parser.metavars['libraries'],) if 'libraries' in parser.metavars else ()
            no_target = parser.metavars.get('no_target',False) or \
                    parser.metavars.get('ignore_at_import',False)
            deps[mod_base] = CompileObj(mod_base,
                                folder          = mod_folder,
                                libs            = compile_libs,
                                has_target_file = not no_target)

        # Proceed recursively
        for son in parser.sons:
            get_module_dependencies(son, deps)

    for son in parser.sons:
        get_module_dependencies(son, deps)
    mod_obj.add_dependencies(*deps.values())

    # Compile code to modules
    try:
        src_compiler.compile_module(compile_obj=mod_obj,
                output_folder=pyccel_dirpath,
                verbose=verbose)
    except Exception:
        handle_error('Fortran compilation')
        raise


    try:
        if codegen.is_program:
            prog_obj = CompileObj(file_name = prog_name,
                    folder       = pyccel_dirpath,
                    dependencies = (mod_obj,),
                    prog_target  = module_name)
            generated_program_filepath = src_compiler.compile_program(compile_obj=prog_obj,
                    output_folder=pyccel_dirpath,
                    verbose=verbose)
        # Create shared library
        generated_filepath = create_shared_library(codegen,
                                               mod_obj,
                                               language,
                                               wrapper_flags,
                                               pyccel_dirpath,
                                               src_compiler,
                                               wrapper_compiler,
                                               output_name,
                                               verbose)
    except NotImplementedError as error:
        msg = str(error)
        errors.report(msg+'\n'+PYCCEL_RESTRICTION_TODO,
            severity='error')
        handle_error('code generation (wrapping)')
        raise PyccelCodegenError(msg) from None
    except PyccelError:
        handle_error('code generation (wrapping)')
        raise
    except Exception:
        handle_error('shared library generation')
        raise

    if errors.has_errors():
        handle_error('code generation (wrapping)')
        raise PyccelCodegenError('Code generation failed')

    # Move shared library to folder directory
    # (First construct absolute path of target location)
    generated_filename = os.path.basename(generated_filepath)
    target = os.path.join(folder, generated_filename)
    shutil.move(generated_filepath, target)
    generated_filepath = target
    if verbose:
        print( '> Shared library has been created: {}'.format(generated_filepath))

    if codegen.is_program:
        generated_program_filename = os.path.basename(generated_program_filepath)
        target = os.path.join(folder, generated_program_filename)
        shutil.move(generated_program_filepath, target)
        generated_program_filepath = target

        if verbose:
            print( '> Executable has been created: {}'.format(generated_program_filepath))

    # Print all warnings now
    if errors.has_warnings():
        errors.check()

    # Change working directory back to starting point
    os.chdir(base_dirpath)
    pyccel_stage.pyccel_finished()
