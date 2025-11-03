# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#

import os
import time

from pyccel.codegen.compiling.basic              import CompileObj
from pyccel.codegen.wrappergen  import Wrappergen
from pyccel.codegen.utilities                    import manage_dependencies
from pyccel.errors.errors                        import Errors

errors = Errors()

__all__ = ['create_shared_library']

#==============================================================================
def create_shared_library(codegen,
                          main_obj,
                          *,
                          language,
                          wrapper_flags,
                          pyccel_dirpath,
                          output_dirpath,
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

    output_dirpath : str
        Path to the directory where the shared library should be outputted.

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

    # Get module name
    module_name = codegen.name

    # Name of shared library
    if sharedlib_modname is None:
        sharedlib_modname = module_name

    gen = Wrappergen(codegen, module_name, language, verbose)

    #-------------------------------------------
    #               Wrap code
    #-------------------------------------------

    start_wrapper_creation = time.time()
    gen.wrap(os.path.dirname(pyccel_dirpath))
    timings['Wrapper creation'] = time.time() - start_wrapper_creation

    if errors.has_errors():
        return '', {}

    #-------------------------------------------
    #           Print wrapper code
    #-------------------------------------------

    start_wrapper_printing = time.time()
    wrapper_files = gen.print(pyccel_dirpath)
    timings['Wrapper printing'] = time.time() - start_wrapper_printing

    if errors.has_errors():
        return

    printed_languages = gen.printed_languages

    #-------------------------------------------
    #         Prepare the compile objects
    #-------------------------------------------

    wrapper_compile_objs = [CompileObj(filepath,
                                       pyccel_dirpath,
                                       flags        = main_obj.flags,
                                       dependencies = (main_obj,))
                            for filepath in wrapper_files[:-1]] + \
                           [CompileObj(wrapper_files[-1],
                                       pyccel_dirpath,
                                       flags        = wrapper_flags,
                                       dependencies = (main_obj,),
                                       extra_compilation_tools = ('python',))]

    for i, (obj, lang, imports) in enumerate(zip(wrapper_compile_objs, printed_languages, gen.get_additional_imports())):
        obj.add_dependencies(*wrapper_compile_objs[:i])
        manage_dependencies(imports, compiler,
                pyccel_dirpath, obj, lang, verbose)

    #-------------------------------------------
    #               Compile code
    #-------------------------------------------

    start_compile_wrapper = time.time()
    for obj, wrapper_language in zip(wrapper_compile_objs, printed_languages):
        compiler.compile_module(compile_obj=obj,
                output_folder=pyccel_dirpath,
                language=wrapper_language,
                verbose=verbose)

    sharedlib_filepath = compiler.compile_shared_library(wrapper_compile_objs[-1],
                                                    output_folder = output_dirpath,
                                                    sharedlib_modname = sharedlib_modname,
                                                    language = language,
                                                    verbose = verbose)

    timings['Wrapper compilation'] = time.time() - start_compile_wrapper

    # Return absolute path of shared library
    return sharedlib_filepath, timings
