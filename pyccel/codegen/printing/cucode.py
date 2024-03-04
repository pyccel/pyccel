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
from itertools import chain

from pyccel.codegen.printing.ccode import CCodePrinter, c_library_headers

from pyccel.ast.core        import Import, Module
from pyccel.ast.core      import SeparatorComment
from pyccel.ast.core      import Declare
from pyccel.ast.core      import FuncAddressDeclare
from pyccel.ast.core      import Assign

from pyccel.errors.errors   import Errors

from pyccel.ast.variable import Variable

from pyccel.ast.literals  import Nil

from pyccel.ast.c_concepts import ObjectAddress

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
                    extern "C"{{\n\
                    {local_imports}\
                    }}'

        code = f'{imports}\n\
                 {global_variables}\n\
                 {body}\n'

        self.exit_scope()
        return code

    def _print_FunctionDef(self, expr):
        if expr.is_inline:
            return ''

        self.set_scope(expr.scope)

        arguments = [a.var for a in expr.arguments]
        results = [r.var for r in expr.results]
        if len(expr.results) > 1:
            self._additional_args.append(results)

        body  = self._print(expr.body)
        decs  = [Declare(i) if isinstance(i, Variable) else FuncAddressDeclare(i) for i in expr.local_vars]

        if len(results) == 1 :
            res = results[0]
            if isinstance(res, Variable) and not res.is_temp:
                decs += [Declare(res)]
            elif not isinstance(res, Variable):
                raise NotImplementedError(f"Can't return {type(res)} from a function")
        decs += [Declare(v) for v in self.scope.variables.values() \
                if v not in chain(expr.local_vars, results, arguments)]
        decs  = ''.join(self._print(i) for i in decs)

        sep = self._print(SeparatorComment(40))
        if self._additional_args :
            self._additional_args.pop()
        for i in expr.imports:
            self.add_import(i)
        docstring = self._print(expr.docstring) if expr.docstring else ''
        cuda_decorater = ''
        if 'kernel' in expr.decorators:
            cuda_decorater = "__global__ "
        if 'device' in expr.decorators:
            cuda_decorater = "__device__ "
        parts = [sep,
                 cuda_decorater,
                 docstring,
                 f'{self.function_signature(expr)}\n{{\n',
                 decs,
                 body,
                 '}\n',
                 sep]

        self.exit_scope()

        return ''.join(p for p in parts if p)

    def _print_KernelCall(self, expr):
        func = expr.funcdef
        if func.is_inline:
            return self._handle_inline_func_call(expr)
        args = []
        for a, f in zip(expr.args, func.arguments):
            arg_val = a.value or Nil()
            f = f.var
            if self.is_c_pointer(f):
                if isinstance(arg_val, Variable):
                    args.append(ObjectAddress(arg_val))
                elif not self.is_c_pointer(arg_val):
                    tmp_var = self.scope.get_temporary_variable(f.dtype)
                    assign = Assign(tmp_var, arg_val)
                    self._additional_code += self._print(assign)
                    args.append(ObjectAddress(tmp_var))
                else:
                    args.append(arg_val)
            else :
                args.append(arg_val)

        args += self._temporary_args
        self._temporary_args = []
        args = ', '.join([f'{self._print(a)}' for a in args])
        return f"{func.name}<<<{expr.numBlocks}, {expr.tpblock}>>>({args});\n"
