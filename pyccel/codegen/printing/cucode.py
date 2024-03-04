# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
Provide tools for generating and handling CUDA code.
This module is designed to interface Pyccel's Abstract Syntax Tree (AST) with CUDA,
enabling the direct translation of high-level Pyccel expressions into CUDA code.
"""

import functools

from pyccel.codegen.printing.ccode import CCodePrinter, c_library_headers, c_imports
from pyccel.ast.datatypes import NativeInteger, NativeVoid

from pyccel.ast.core      import Deallocate
from pyccel.ast.variable import DottedVariable

from pyccel.ast.core        import Import, Module, Declare
from pyccel.ast.operators import PyccelMul

from pyccel.errors.errors   import Errors


errors = Errors()

__all__ = ["CudaCodePrinter"]

class CudaCodePrinter(CCodePrinter):
    """
    Print code in CUDA format.

    This printer converts Pyccel's Abstract Syntax Tree (AST) into strings of CUDA code.
    Navigation through this file utilizes _print_X functions,
    as is common with all printers.

    Parameters
    ----------
    filename : str
            The name of the file being pyccelised.
    prefix_module : str
            A prefix to be added to the name of the module.
    """
    language = "cuda"

    def __init__(self, filename, prefix_module = None):

        errors.set_target(filename, 'file')

        super().__init__(filename)

    def _print_Program(self, expr):
        self.set_scope(expr.scope)
        body  = self._print(expr.body)
        variables = self.scope.variables.values()
        decs = ''.join(self._print(Declare(v)) for v in variables)

        imports = [*expr.imports, *self._additional_imports.values()]
        c_headers_imports = ''
        local_imports = ''

        for imp in imports:
            if imp.source in c_library_headers:
                c_headers_imports += self._print(imp)
            else:
                local_imports += self._print(imp)

        imports = f'{c_headers_imports}\
                    extern "C"{{\n\
                    {local_imports}\
                    }}'
        # imports = ''.join(self._print(i) for i in imports)

        self.exit_scope()
        return ('{imports}'
                'int main()\n{{\n'
                '{decs}'
                '{body}'
                'return 0;\n'
                '}}').format(imports=imports,
                                    decs=decs,
                                    body=body)

    def _print_Module(self, expr):
        self.set_scope(expr.scope)
        self._current_module = expr.name
        body = ''.join(self._print(i) for i in expr.body)

        global_variables = ''.join(self._print(d) for d in expr.declarations)

        # Print imports last to be sure that all additional_imports have been collected
        imports = [Import(expr.name, Module(expr.name,(),())), *self._additional_imports.values()]
        c_headers_imports = ''
        local_imports = ''

        for imp in imports:
            if imp.source in c_library_headers:
                c_headers_imports += self._print(imp)
            else:
                local_imports += self._print(imp)

        imports = f'{c_headers_imports}\
                    extern "C"{{\n\
                    {local_imports}\
                    }}'

        code = f'{imports}\n\
                 {global_variables}\n\
                 {body}\n'

        self.exit_scope()
        return code

    def _init_stack_array(self, expr):
        """
        Return a string which handles the assignment of a stack ndarray.

        Print the code necessary to initialise a ndarray on the stack.

        Parameters
        ----------
        expr : TypedAstNode
            The Assign Node used to get the lhs and rhs.

        Returns
        -------
        buffer_array : str
            String initialising the stack (C) array which stores the data.
        array_init   : str
            String containing the rhs of the initialization of a stack array.
        """
        var = expr
        dtype = self.find_in_dtype_registry(var.dtype, var.precision)
        np_dtype = self.find_in_ndarray_type_registry(var.dtype, var.precision)
        shape = ", ".join(self._print(i) for i in var.alloc_shape)
        tot_shape = self._print(functools.reduce(
            lambda x,y: PyccelMul(x,y,simplify=True), var.alloc_shape))
        declare_dtype = self.find_in_dtype_registry(NativeInteger(), 8)

        dummy_array_name = self.scope.get_new_name('array_dummy')
        buffer_array = "{dtype} {name}[{size}];\n".format(
                dtype = dtype,
                name  = dummy_array_name,
                size  = tot_shape)
        tmp_shape = self.scope.get_new_name(f'tmp_shape_{var.name}')
        shape_init = f'{declare_dtype} {tmp_shape}[] = {{{shape}}};\n'
        tmp_strides = self.scope.get_new_name(f'tmp_strides_{var.name}')
        strides_init = f'{declare_dtype} {tmp_strides}[{var.rank}] = {{0}};\n'
        array_init = f' = (t_ndarray){{\n.{np_dtype}={dummy_array_name},\n .nd={var.rank},\n '
        array_init += f'.shape={tmp_shape},\n .strides={tmp_strides},\n .type={np_dtype},\n .is_view=false\n}};\n'
        array_init += 'stack_array_init(&{})'.format(self._print(var))
        preface = buffer_array + shape_init + strides_init
        self.add_import(c_imports['ndarrays'])
        return preface, array_init

    def _print_Allocate(self, expr):
        free_code = ''
        variable = expr.variable
        if variable.rank > 0:
            #free the array if its already allocated and checking if its not null if the status is unknown
            if  (expr.status == 'unknown'):
                shape_var = DottedVariable(NativeVoid(), 'shape', lhs = variable)
                free_code = f'if ({self._print(shape_var)} != NULL)\n'
                free_code += "{{\n{}}}\n".format(self._print(Deallocate(variable)))
            elif (expr.status == 'allocated'):
                free_code += self._print(Deallocate(variable))
            self.add_import(c_imports['ndarrays'])
            shape = ", ".join(self._print(i) for i in expr.shape)
            dtype = self.find_in_ndarray_type_registry(variable.dtype, variable.precision)
            shape_dtype = self.find_in_dtype_registry(NativeInteger(), 8)
            tmp_shape = self.scope.get_new_name(f'tmp_shape_{self._print(variable)}')
            shape_Assign = f'{shape_dtype} {tmp_shape}[] = {{{shape}}};\n'
            is_view = 'false' if variable.on_heap else 'true'
            order = "order_f" if expr.order == "F" else "order_c"
            alloc_code = f"{self._print(variable)} = array_create({variable.rank}, {tmp_shape}, {dtype}, {is_view}, {order});\n"
            return '{}{}{}'.format(free_code, shape_Assign,alloc_code)
        elif variable.is_alias:
            var_code = self._print(ObjectAddress(variable))
            if expr.like:
                declaration_type = self.get_declare_type(expr.like)
                return f'{var_code} = malloc(sizeof({declaration_type}));\n'
            else:
                raise NotImplementedError(f"Allocate not implemented for {variable}")
        else:
            raise NotImplementedError(f"Allocate not implemented for {variable}")