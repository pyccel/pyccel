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

from pyccel.codegen.printing.ccode  import CCodePrinter

from pyccel.ast.core                import Import, Module
from pyccel.ast.literals            import Nil

from pyccel.errors.errors           import Errors
from pyccel.ast.core                import Allocate, Deallocate



errors = Errors()

__all__ = ["CudaCodePrinter"]

c_imports = {n : Import(n, Module(n, (), ())) for n in
                ['cuda_ndarrays',]}

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

        errors.set_target(filename)

        super().__init__(filename)

    def _print_Module(self, expr):
        self.set_scope(expr.scope)
        self._current_module = expr.name
        body = ''.join(self._print(i) for i in expr.body)

        global_variables = ''.join(self._print(d) for d in expr.declarations)

        # Print imports last to be sure that all additional_imports have been collected
        imports = [Import(expr.name, Module(expr.name,(),())), *self._additional_imports.values()]
        imports = ''.join(self._print(i) for i in imports)

        code = f'{imports}\n\
                 {global_variables}\n\
                 {body}\n'

        self.exit_scope()
        return code

    def function_signature(self, expr, print_arg_names = True):
        """
        Get the Cuda representation of the function signature.

        Extract from the function definition `expr` all the
        information (name, input, output) needed to create the
        function signature and return a string describing the
        function.
        This is not a declaration as the signature does not end
        with a semi-colon.

        Parameters
        ----------
        expr : FunctionDef
            The function definition for which a signature is needed.

        print_arg_names : bool, default : True
            Indicates whether argument names should be printed.

        Returns
        -------
        str
            Signature of the function.
        """
        cuda_decorater = '__global__' if 'kernel' in expr.decorators else ''
        c_function_signature = super().function_signature(expr, print_arg_names)
        return f'{cuda_decorater} {c_function_signature}'

    def _print_KernelCall(self, expr):
        func = expr.funcdef
        args = [a.value or Nil() for a in expr.args]

        args = ', '.join(self._print(a) for a in args)
        return f"{func.name}<<<{expr.num_blocks}, {expr.tp_block}>>>({args});\n"

    def _print_CudaSynchronize(self, expr):
        return 'cudaDeviceSynchronize();\n'

    def _print_CudaEmpty(self, expr):
        print(expr)
        return 'cudaDeviceSynchronize();\n'
    def _print_ModuleHeader(self, expr):
        self.set_scope(expr.module.scope)
        self._in_header = True
        name = expr.module.name

        funcs = ""
        cuda_headers = ""
        for f in expr.module.funcs:
            if not f.is_inline:
                if 'kernel' in f.decorators:  # Checking for 'kernel' decorator
                    cuda_headers += self.function_signature(f) + ';\n'
                else:
                    funcs += self.function_signature(f) + ';\n'
        global_variables = ''.join('extern '+self._print(d) for d in expr.module.declarations if not d.variable.is_private)
        # Print imports last to be sure that all additional_imports have been collected
        imports = [*expr.module.imports, *self._additional_imports.values()]
        imports = ''.join(self._print(i) for i in imports)

        self._in_header = False
        self.exit_scope()
        function_declaration = f'{cuda_headers}\n\
                    extern "C"{{\n\
                    {funcs}\
                    }}\n'
        return '\n'.join((f"#ifndef {name.upper()}_H",
                          f"#define {name.upper()}_H",
                          imports,
                          global_variables,
                          function_declaration,
                          "#endif // {name.upper()}_H\n"))
    def _print_Allocate(self, expr):
 
        self.add_import(c_imports['cuda_ndarrays'])
        alloc_code = f"{self._print(expr.variable)} = cuda_array_create();\n"
        return f'{alloc_code}'
        # print(shape)
        
        # return "hjsjkahsjkajskasjkasj"

