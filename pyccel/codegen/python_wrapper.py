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
    wrapper_code = wrapper_codegen.doprint(c_ast)
    if errors.has_errors():
        return

    codegen.ast.set_name(module_old_name)

    with open(wrapper_filename, 'w') as f:
        f.writelines(wrapper_code)

    #--------------------------------------------------------
    #  Compile cwrapper_ndarrays from stdlib (if necessary)
    #--------------------------------------------------------
    if "ndarrays" in wrapper_codegen.get_additional_imports():
        for lib_name in ("ndarrays", "cwrapper_ndarrays"):
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
