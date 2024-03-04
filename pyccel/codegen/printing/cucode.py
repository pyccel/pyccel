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

from pyccel.codegen.printing.ccode import CCodePrinter, c_library_headers, c_imports
from pyccel.ast.datatypes import NativeInteger, NativeVoid

from pyccel.ast.core      import Deallocate
from pyccel.ast.variable import DottedVariable

from pyccel.ast.core        import Import, Module, Declare

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