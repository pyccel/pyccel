# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

from pyccel.codegen.printing.ccode import CCodePrinter, c_imports

from pyccel.ast.variable    import InhomogeneousTupleVariable
from pyccel.ast.core        import Declare, Import, Module

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

__all__ = ["CudaCodePrinter", "cucode"]

class CudaCodePrinter(CCodePrinter):
    """
    A printer for printing code in Cuda.

    A printer to convert Pyccel's AST to strings of cuda code.
    As for all printers the navigation of this file is done via _print_X
    functions.

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

def cucode(expr, filename, assign_to=None, **settings):
    """Converts an expr to a string of cuda code

    expr : Expr
        A pyccel expression to be converted.
    filename : str
        The name of the file being translated. Used in error printing
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned. Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of
        line-wrapping, or for expressions that generate multi-line statements.
    precision : integer, optional
        The precision for numbers such as pi [default=15].
    user_functions : dict, optional
        A dictionary where keys are ``FunctionClass`` instances and values are
        their string representations. Alternatively, the dictionary value can
        be a list of tuples i.e. [(argument_test, cfunction_string)]. See below
        for examples.
    dereference : iterable, optional
        An iterable of symbols that should be dereferenced in the printed code
        expression. These would be values passed by address to the function.
        For example, if ``dereference=[a]``, the resulting code would print
        ``(*a)`` instead of ``a``.
    """
    return CudaCodePrinter(filename, **settings).doprint(expr, assign_to)
