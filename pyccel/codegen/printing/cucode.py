# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

from pyccel.codegen.printing.ccode import CCodePrinter, c_imports

from pyccel.ast.c_concepts  import ObjectAddress
from pyccel.ast.datatypes   import NativeInteger, NativeVoid
from pyccel.ast.variable    import DottedVariable, DottedName
from pyccel.ast.core        import Deallocate, AsName, Import, Module


from pyccel.errors.errors   import Errors
from pyccel.errors.messages import (PYCCEL_RESTRICTION_TODO, INCOMPATIBLE_TYPEVAR_TO_FUNC,
                                    PYCCEL_RESTRICTION_IS_ISNOT, UNSUPPORTED_ARRAY_RANK)


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

        code = ('{imports}\n'
                '{variables}\n'
                '{body}\n').format(
                        imports   = imports,
                        variables = global_variables,
                        body      = body)

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
            tmp_shape = self.scope.get_new_name('tmp_shape')
            shape_Assign = "{} {}[] = {{{}}};".format(shape_dtype, tmp_shape, shape)
            is_view = 'false' if variable.on_heap else 'true'
            order = "order_f" if expr.order == "F" else "order_c"
            alloc_code = f"{self._print(variable)} = array_create({variable.rank}, {tmp_shape}, {dtype}, {is_view}, {order});\n"
            return f'{free_code}\n{shape_Assign}\n{alloc_code}'
        elif variable.is_alias:
            var_code = self._print(ObjectAddress(variable))
            if expr.like:
                declaration_type = self.get_declare_type(expr.like)
                return f'{var_code} = malloc(sizeof({declaration_type}));\n'
            else:
                raise NotImplementedError(f"Allocate not implemented for {variable}")
        else:
            raise NotImplementedError(f"Allocate not implemented for {variable}")


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
