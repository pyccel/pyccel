# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

import os

from pyccel.ast.numpy_wrapper               import get_numpy_max_acceptable_version_file
from pyccel.codegen.printing.fcode          import fcode
from pyccel.codegen.printing.cwrappercode   import CWrapperCodePrinter
from pyccel.codegen.wrapper.fortran_to_c_wrapper   import FortranToCWrapper
from pyccel.codegen.wrapper.c_to_python_wrapper    import CToPythonWrapper
from pyccel.codegen.utilities      import recompile_object
from pyccel.codegen.utilities      import copy_internal_library
from pyccel.codegen.utilities      import internal_libs
from pyccel.naming                 import name_clash_checkers
from pyccel.parser.scope           import Scope
from pyccel.utilities.stage        import PyccelStage
from .compiling.basic     import CompileObj

from pyccel.errors.errors import Errors

errors = Errors()

pyccel_stage = PyccelStage()

__all__ = ['create_shared_library']

#==============================================================================
def create_shared_library(codegen,
                          main_obj,
                          language,
                          wrapper_flags,
                          pyccel_dirpath,
                          src_compiler,
                          wrapper_compiler,
                          sharedlib_modname=None,
                          verbose = False):
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

    src_compiler : pyccel.codegen.compiling.compilers.Compiler
        The compiler which should be used to compile the library.

    wrapper_compiler : pyccel.codegen.compiling.compilers.Compiler
        The compiler which should be used to compile the wrapper.
        Often this is the same as src_compiler but it may be different
        when the language is not C to ensure that src_compiler can link
        the appropriate language-specific libraries.

    sharedlib_modname : str, default: None
        The name of the shared library. The default is the name of the
        module printed by the printer.

    verbose : bool, default: False
        Indicates if the compiling should be done with verbosity to show the
        compiler commands.

    Returns
    -------
    str
        The absolute path to the shared library which was created.
    """

    pyccel_stage.set_stage('cwrapper')

    # Get module name
    module_name = codegen.name

    # Change working directory to '__pyccel__'
    base_dirpath = os.getcwd()
    os.chdir(pyccel_dirpath)

    # Name of shared library
    if sharedlib_modname is None:
        sharedlib_modname = module_name

    wrapper_filename_root = '{}_wrapper'.format(module_name)
    wrapper_filename = '{}.c'.format(wrapper_filename_root)
    wrapper_compile_obj = CompileObj(wrapper_filename,
            pyccel_dirpath,
            flags        = wrapper_flags,
            dependencies = (main_obj,),
            accelerators = ('python',))

    if language == 'fortran':
        # Construct static interface for passing array shapes and write it to file bind_c_MOD.f90
        wrapper = FortranToCWrapper()
        bind_c_mod = wrapper.wrap(codegen.ast)
        bind_c_code = fcode(bind_c_mod, bind_c_mod.name)
        bind_c_filename = '{}.f90'.format(bind_c_mod.name)

        with open(bind_c_filename, 'w') as f:
            f.writelines(bind_c_code)

        bind_c_obj=CompileObj(file_name = bind_c_filename,
                folder = pyccel_dirpath,
                flags  = main_obj.flags,
                dependencies = (main_obj,))
        wrapper_compile_obj.add_dependencies(bind_c_obj)
        src_compiler.compile_module(compile_obj=bind_c_obj,
                output_folder=pyccel_dirpath,
                verbose=verbose)
        c_ast = bind_c_mod
    else:
        c_ast = codegen.ast

    #---------------------------------------
    #     Compile cwrapper from stdlib
    #---------------------------------------
    cwrapper_lib_dest_path = copy_internal_library('cwrapper', pyccel_dirpath,
                                extra_files = {'numpy_version.h' :
                                                get_numpy_max_acceptable_version_file()})

    cwrapper_lib = internal_libs["cwrapper"][1]
    cwrapper_lib.reset_folder(cwrapper_lib_dest_path)

    # get the include folder path and library files
    recompile_object(cwrapper_lib,
                      compiler = wrapper_compiler,
                      verbose  = verbose)

    wrapper_compile_obj.add_dependencies(cwrapper_lib)

    #---------------------------------------
    #      Print code specific cwrapper
    #---------------------------------------
    module_old_name = codegen.ast.name
    codegen.ast.set_name(sharedlib_modname)
    wrapper_codegen = CWrapperCodePrinter(codegen.parser.filename, language)
    Scope.name_clash_checker = name_clash_checkers['c']
    wrapper = CToPythonWrapper()
    cwrap_ast = wrapper.wrap(c_ast)
    wrapper_code = wrapper_codegen.doprint(cwrap_ast)
    #wrapper_code = wrapper_codegen.doprint(c_ast)
    if errors.has_errors():
        return

    codegen.ast.set_name(module_old_name)

    with open(wrapper_filename, 'w') as f:
        f.writelines(wrapper_code)

    #--------------------------------------------------------
    #  Compile cwrapper_ndarrays from stdlib (if necessary)
    #--------------------------------------------------------
    for lib_name in ("ndarrays", "cwrapper_ndarrays"):
        if lib_name in wrapper_codegen.get_additional_imports():
            stdlib_folder, stdlib = internal_libs[lib_name]

            lib_dest_path = copy_internal_library(stdlib_folder, pyccel_dirpath)

            # Pylint determines wrong type
            stdlib.reset_folder(lib_dest_path) # pylint: disable=E1101
            # get the include folder path and library files
            recompile_object(stdlib,
                              compiler = wrapper_compiler,
                              verbose  = verbose)

            wrapper_compile_obj.add_dependencies(stdlib)

    #---------------------------------------
    #         Compile code
    #---------------------------------------
    wrapper_compiler.compile_module(wrapper_compile_obj,
                                output_folder = pyccel_dirpath,
                                verbose = verbose)

    sharedlib_filepath = src_compiler.compile_shared_library(wrapper_compile_obj,
                                                    output_folder = pyccel_dirpath,
                                                    sharedlib_modname = sharedlib_modname,
                                                    verbose = verbose)

    # Change working directory back to starting point
    os.chdir(base_dirpath)

    # Return absolute path of shared library
    return sharedlib_filepath
