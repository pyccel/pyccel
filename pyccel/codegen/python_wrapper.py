# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#

import os
import time

from pyccel.ast.core                             import ModuleHeader
from pyccel.codegen.compiling.basic              import CompileObj
from pyccel.codegen.printing.cwrappercode        import CWrapperCodePrinter
from pyccel.codegen.printing.fcode               import FCodePrinter
from pyccel.codegen.wrapper.fortran_to_c_wrapper import FortranToCWrapper
from pyccel.codegen.wrapper.c_to_python_wrapper  import CToPythonWrapper
from pyccel.codegen.utilities                    import manage_dependencies
from pyccel.errors.errors                        import Errors
from pyccel.naming                               import name_clash_checkers
from pyccel.parser.scope                         import Scope
from pyccel.utilities.stage                      import PyccelStage

errors = Errors()

pyccel_stage = PyccelStage()

__all__ = ['create_shared_library']

#==============================================================================
def create_shared_library(codegen,
                          main_obj,
                          *,
                          language,
                          wrapper_flags,
                          pyccel_dirpath,
                          compiler,
                          sharedlib_modname=None,
                          verbose):
    """
    Create a shared library which can be called from Pyccel.

    From a CodePrinter object describing code which has been printed
    in a target language, create a shared library which can be
    called from Pyccel. In order to do this the code must be wrapped.
    First, if the code is not written in C, it must be wrapped to
    make it callable from C. This intermediary code is printed
    and compiled. From the C-compatible code a second (first for C)
    wrapper is created which exposes the C code to Python. This
    is done via the CWrapper. Finally this new code is compiled
    to generate the required shared language.

    Parameters
    ----------
    codegen : pyccel.codegen.printing.codeprinter.CodePrinter
        The printer which was used to print the translated code.

    main_obj : pyccel.codegen.compiling.basic.CompileObj
        The compile object which describes the translated code.

    language : str
        The language which Pyccel translated to.

    wrapper_flags : iterable
        Any additional flags which should be used to compile the wrapper.

    pyccel_dirpath : str
        The path to the directory where the files are created and compiled.

    compiler : pyccel.codegen.compiling.compilers.Compiler
        The compiler which should be used to compile the library.

    sharedlib_modname : str, default: None
        The name of the shared library. The default is the name of the
        module printed by the printer.

    verbose : int
        Indicates the level of verbosity.

    Returns
    -------
    sharedlib_filepath : str
        The absolute path to the shared library which was created.

    timings : dict
        The time spent in the different parts of the library creation.
    """
    timings = {}

    pyccel_stage.set_stage('cwrapper')

    # Get module name
    module_name = codegen.name

    # Change working directory to '__pyccel__'
    base_dirpath = os.getcwd()
    os.chdir(pyccel_dirpath)

    # Name of shared library
    if sharedlib_modname is None:
        sharedlib_modname = module_name

    wrapper_filename_root = f'{module_name}_wrapper'
    wrapper_header_filename = f'{wrapper_filename_root}.h'
    wrapper_filename = f'{wrapper_filename_root}.c'
    wrapper_compile_obj = CompileObj(wrapper_filename,
            pyccel_dirpath,
            flags        = wrapper_flags,
            dependencies = (main_obj,),
            accelerators = ('python',))

    if language == 'fortran':
        start_bind_c_wrapping = time.time()
        if verbose:
            print(">> Building Fortran-C interface :: ", module_name)
        # Construct static interface for passing array shapes and write it to file bind_c_MOD.f90
        wrapper = FortranToCWrapper(verbose)
        bind_c_mod = wrapper.wrap(codegen.ast)
        timings['Bind C wrapping'] = time.time() - start_bind_c_wrapping

        bind_c_filename = f'{bind_c_mod.name}.f90'
        if verbose:
            print(">> Printing :: ", bind_c_filename)
        start_bind_c_printing = time.time()
        bind_c_code = FCodePrinter(bind_c_mod.name, verbose=verbose).doprint(bind_c_mod)

        with open(bind_c_filename, 'w') as f:
            f.writelines(bind_c_code)
        timings['Bind C printing'] = time.time() - start_bind_c_printing

        start_bind_c_compiling = time.time()
        bind_c_obj=CompileObj(file_name = bind_c_filename,
                folder = pyccel_dirpath,
                flags  = main_obj.flags,
                dependencies = (main_obj,))
        wrapper_compile_obj.add_dependencies(bind_c_obj)
        compiler.compile_module(compile_obj=bind_c_obj,
                output_folder=pyccel_dirpath,
                language=language,
                verbose=verbose)
        timings['Bind C wrapping'] = time.time() - start_bind_c_compiling
        c_ast = bind_c_mod
    else:
        c_ast = codegen.ast

    #---------------------------------------
    #      Print code specific cwrapper
    #---------------------------------------
    wrapper_codegen = CWrapperCodePrinter(codegen.parser.filename, language, verbose=verbose)
    Scope.name_clash_checker = name_clash_checkers['c']
    wrapper = CToPythonWrapper(base_dirpath, verbose)

    if verbose:
        print(">> Building C-Python interface :: ", c_ast.name)
    start_wrapper_creation = time.time()
    cwrap_ast = wrapper.wrap(c_ast)
    timings['Wrapper creation'] = time.time() - start_wrapper_creation

    if verbose:
        print(">> Printing :: ", wrapper_filename)
    start_print_cwrapper = time.time()
    wrapper_code = wrapper_codegen.doprint(cwrap_ast)
    #wrapper_code = wrapper_codegen.doprint(c_ast)
    if errors.has_errors():
        return

    with open(wrapper_filename, 'w', encoding="utf-8") as f:
        f.writelines(wrapper_code)
    timings['Wrapper printing'] = time.time() - start_print_cwrapper

    wrapper_header_code = wrapper_codegen.doprint(ModuleHeader(cwrap_ast))

    with open(wrapper_header_filename, 'w', encoding="utf-8") as f:
        f.writelines(wrapper_header_code)

    #--------------------------------------------------------
    #  Compile cwrapper_ndarrays from stdlib (if necessary)
    #--------------------------------------------------------
    start_compile_libs = time.time()
    manage_dependencies(wrapper_codegen.get_additional_imports(), compiler,
            pyccel_dirpath, wrapper_compile_obj, 'c', verbose)
    timings['Dependency compilation'] = (time.time() - start_compile_libs)

    #---------------------------------------
    #         Compile code
    #---------------------------------------
    start_compile_wrapper = time.time()
    compiler.compile_module(wrapper_compile_obj,
                            output_folder = pyccel_dirpath,
                            language='c',
                            verbose = verbose)

    sharedlib_filepath = compiler.compile_shared_library(wrapper_compile_obj,
                                                    output_folder = pyccel_dirpath,
                                                    sharedlib_modname = sharedlib_modname,
                                                    language = language,
                                                    verbose = verbose)
    timings['Wrapper compilation'] = time.time() - start_compile_wrapper

    # Change working directory back to starting point
    os.chdir(base_dirpath)

    # Return absolute path of shared library
    return sharedlib_filepath, timings
