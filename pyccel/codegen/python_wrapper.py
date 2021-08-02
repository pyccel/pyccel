# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

import os

from pyccel.ast.bind_c                      import as_static_module
from pyccel.codegen.printing.fcode          import fcode
from pyccel.codegen.printing.cwrappercode   import cwrappercode
from .compiling.basic     import CompileObj

from pyccel.errors.errors import Errors

errors = Errors()

__all__ = ['create_shared_library', 'fortran_c_flag_equivalence']

#==============================================================================

fortran_c_flag_equivalence = {'-Wconversion-extra' : '-Wconversion' }

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

    # Consistency checks
    if not codegen.is_module:
        raise TypeError('Expected Module')

    # Get module name
    module_name = codegen.name

    # Change working directory to '__pyccel__'
    base_dirpath = os.getcwd()
    os.chdir(pyccel_dirpath)

    # Name of shared library
    if sharedlib_modname is None:
        sharedlib_modname = module_name

    sharedlib_folder = ''

    wrapper_filename_root = '{}_wrapper'.format(module_name)
    wrapper_filename = '{}.c'.format(wrapper_filename_root)
    wrapper_compile_obj = CompileObj(wrapper_filename,
            pyccel_dirpath,
            flags        = wrapper_flags,
            dependencies = (main_obj,),
            accelerators = ('python',))

    if language == 'fortran':
        # Construct static interface for passing array shapes and write it to file bind_c_MOD.f90
        new_module_name = 'bind_c_{}'.format(module_name)
        bind_c_mod = as_static_module(codegen.routines, module_name, new_module_name)
        bind_c_code = fcode(bind_c_mod, codegen.parser)
        bind_c_filename = '{}.f90'.format(new_module_name)

        with open(bind_c_filename, 'w') as f:
            f.writelines(bind_c_code)

        bind_c_obj=CompileObj(file_name = bind_c_filename,
                folder = pyccel_dirpath,
                flags  = main_obj.flags,
                is_module = True,
                dependencies = (main_obj,))
        wrapper_compile_obj.add_dependencies(bind_c_obj)
        src_compiler.compile_module(compile_obj=bind_c_obj,
                output_folder=pyccel_dirpath,
                verbose=verbose)

    module_old_name = codegen.expr.name
    codegen.expr.set_name(sharedlib_modname)
    wrapper_code = cwrappercode(codegen.expr, codegen.parser, language)
    if errors.has_errors():
        return

    codegen.expr.set_name(module_old_name)

    with open(wrapper_filename, 'w') as f:
        f.writelines(wrapper_code)

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
