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
from pyccel.ast.literals  import Nil

from pyccel.codegen.printing.ccode import CCodePrinter, c_imports

from pyccel.ast.variable    import InhomogeneousTupleVariable
from pyccel.ast.variable import Variable

from pyccel.ast.core        import Declare, Import, Module, Assign
from pyccel.ast.core      import SeparatorComment, FuncAddressDeclare

from pyccel.ast.c_concepts import ObjectAddress


from pyccel.errors.errors   import Errors


import_dict = {'omp_lib' : 'omp' }

c_library_headers = (
    "complex",
    "ctype",
    "float",
    "math",
    "stdarg",
    "stdbool",
    "stddef",
    "stdint",
    "stdio",
    "stdlib",
    "string",
    "tgmath",
    "inttypes",
)

errors = Errors()

# TODO: add examples

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
    printmethod = "_cucode"
    language = "cuda"

    def __init__(self, filename, prefix_module = None):

        errors.set_target(filename, 'file')

        super().__init__(filename)
        self.prefix_module = prefix_module
        self._additional_imports = {'stdlib':c_imports['stdlib']}
        self._additional_code = ''
        self._additional_args = []
        self._temporary_args = []
        self._current_module = None
        self._in_header = False

    def _print_Module(self, expr):
        self.set_scope(expr.scope)
        self._current_module = expr.name
        body    = ''.join(self._print(i) for i in expr.body)

        global_variables = ''.join([self._print(d) for d in expr.declarations])

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

    def _print_Declare(self, expr):
        if isinstance(expr.variable, InhomogeneousTupleVariable):
            return ''.join(self._print_Declare(Declare(v.dtype,v,intent=expr.intent, static=expr.static)) for v in expr.variable)

        declaration_type = self.get_declare_type(expr.variable)
        variable = self._print(expr.variable.name)

        if expr.variable.is_stack_array:
            preface, init = self._init_stack_array(expr.variable,)
        elif declaration_type == 't_ndarray' and not self._in_header:
            preface = ''
            init    = ' = {.shape = NULL}'
        else:
            preface = ''
            init    = ''

        declaration = f'{declaration_type} {variable}{init};\n'

        return preface + declaration
    def _print_FunctionDef(self, expr):
        if expr.is_inline:
            return ''
        self.set_scope(expr.scope)

        # Extract the function arguments and results variables
        arguments = [a.var for a in expr.arguments]
        results = [r.var for r in expr.results]

        # Append the results to additional arguments list if there is more than one result
        if len(expr.results) > 1:
            self._additional_args.append(results)

        # Print the function body
        body = self._print(expr.body)

        # Collect the local variables (excluding function arguments and results)
        decs = [Declare(i.dtype, i) if isinstance(i, Variable) else FuncAddressDeclare(i) for i in expr.local_vars]

        # Declare the single result, if applicable
        if len(results) == 1:
            res = results[0]
            if isinstance(res, Variable) and not res.is_temp:
                decs += [Declare(res.dtype, res)]
            elif not isinstance(res, Variable):
                raise NotImplementedError(f"Can't return {type(res)} from a function")

        # Declare the remaining variables in the function scope
        decs += [Declare(v.dtype,v) for v in self.scope.variables.values() if v not in chain(expr.local_vars, results,  arguments)]

        # Generate the code for the declarations
        decs = ''.join(self._print(i) for i in decs)

        # Print a separator comment
        sep = self._print(SeparatorComment(40))

        # Remove the additional arguments list if it exists
        if self._additional_args:
            self._additional_args.pop()

        # Add the imports to the code
        for i in expr.imports:
            self.add_import(i)

        # Generate the code for the docstring, if exists
        docstring = self._print(expr.docstring) if expr.docstring else ''
    
        # Check if the 'kernel' decorator is present and set the CUDA decorator accordingly
        cuda_decorater = ''
        if 'kernel' in expr.decorators:
            cuda_decorater = "__global__ "

        # Assemble the different parts of the function definition
        parts = [sep,
             cuda_decorater,
             docstring,
             '{signature}\n{{\n'.format(signature=self.function_signature(expr)),
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
         # Ensure the correct syntax is used for pointers
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
        args = ', '.join(['{}'.format(self._print(a)) for a in args])
        
        return f"{func.name}<<<{expr.numBlocks}, {expr.tpblock}>>>({args});\n"
