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

from pyccel.codegen.printing.ccode import CCodePrinter, c_library_headers

from pyccel.ast.core        import Import, Module, AsName

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
                   \n\
                    {local_imports}\
                    '

        code = f'{imports}\n\
                 {global_variables}\n\
                 {body}\n'

        self.exit_scope()
        return code
    def _print_ModuleHeader(self, expr):
        self.set_scope(expr.module.scope)
        self._in_header = True
        name = expr.module.name
        if isinstance(name, AsName):
            name = name.name
        # TODO: Add interfaces
        classes = ""
        funcs = ""
        for classDef in expr.module.classes:
            if classDef.docstring is not None:
                classes += self._print(classDef.docstring)
            classes += f"struct {classDef.name} {{\n"
            classes += ''.join(self._print(Declare(var)) for var in classDef.attributes)
            class_scope = classDef.scope
            for method in classDef.methods:
                if not method.is_inline:
                    class_scope.rename_function(method, f"{classDef.name}__{method.name.lstrip('__')}")
                    funcs += f"{self.function_signature(method)};\n"
            for interface in classDef.interfaces:
                for func in interface.functions:
                    if not func.is_inline:
                        class_scope.rename_function(func, f"{classDef.name}__{func.name.lstrip('__')}")
                        funcs += f"{self.function_signature(func)};\n"
            classes += "};\n"
        cuda_headers = ''
        for f in expr.module.funcs:
            if not f.is_inline:
                if 'kernel' in f.decorators:  # Checking for 'kernel' decorator
                    cuda_headers += self.function_signature(f) + ';'
                else:
                    funcs += self.function_signature(f) + ';'
        global_variables = ''.join(['extern '+self._print(d) for d in expr.module.declarations if not d.variable.is_private])

        # Print imports last to be sure that all additional_imports have been collected
        imports = [*expr.module.imports, *self._additional_imports.values()]
        imports = ''.join(self._print(i) for i in imports)

        self._in_header = False
        self.exit_scope()
        imports = f'{cuda_headers}\n\
                    extern "C"{{\n\
                    {funcs}\n\
                    }}'
        return (f"#ifndef {name.upper()}_H\n \
                #define {name.upper()}_H\n\n \
                {global_variables}\n \
                {classes}\n \
                {imports}\n \
                #endif // {name}_H\n")