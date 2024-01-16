# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

from pyccel.codegen.printing.ccode import CCodePrinter, c_imports

from pyccel.ast.c_concepts  import ObjectAddress
from pyccel.ast.datatypes   import NativeInteger, NativeVoid
from pyccel.ast.variable    import DottedVariable
from pyccel.ast.core        import Deallocate, FunctionAddress, Declare, AsName


from pyccel.errors.errors   import Errors
from pyccel.errors.messages import (PYCCEL_RESTRICTION_TODO, INCOMPATIBLE_TYPEVAR_TO_FUNC,
                                    PYCCEL_RESTRICTION_IS_ISNOT, UNSUPPORTED_ARRAY_RANK)


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

    def function_signature(self, expr, print_arg_names = True, extern_c=True):
        """
        Get the C representation of the function signature.

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
        arg_vars = [a.var for a in expr.arguments]
        result_vars = [r.var for r in expr.results if not r.is_argument]

        n_results = len(result_vars)

        if n_results == 1:
            ret_type = self.get_declare_type(result_vars[0])
        elif n_results > 1:
            ret_type = self.find_in_dtype_registry(NativeInteger(), -1)
            arg_vars.extend(result_vars)
            self._additional_args.append(result_vars) # Ensure correct result for is_c_pointer
        else:
            ret_type = self.find_in_dtype_registry(NativeVoid(), 0)

        name = expr.name
        if not arg_vars:
            arg_code = 'void'
        else:
            def get_arg_declaration(var):
                """ Get the code which declares the argument variable.
                """
                code = "const " * var.is_const
                code += self.get_declare_type(var)
                if print_arg_names:
                    code += ' ' + var.name
                return code

            arg_code_list = [self.function_signature(var, False) if isinstance(var, FunctionAddress)
                                else get_arg_declaration(var) for var in arg_vars]
            arg_code = ', '.join(arg_code_list)

        if self._additional_args :
            self._additional_args.pop()
        extern_c = ''
        # extern_c = 'extern "C" ' if extern_c else ''
        if isinstance(expr, FunctionAddress):
            return f'{extern_c}{ret_type} (*{name})({arg_code})'
        else:
            return f'{extern_c}{ret_type} {name}({arg_code})'

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
            classes += ''.join(self._print(Declare(var.dtype,var)) for var in classDef.attributes)
            for method in classDef.methods:
                if not method.is_inline:
                    method.rename(classDef.name + ("__" + method.name if not method.name.startswith("__") else method.name))
                    funcs += f"{self.function_signature(method, extern_c=False)};\n"
            for interface in classDef.interfaces:
                for func in interface.functions:
                    if not func.is_inline:
                        func.rename(classDef.name + ("__" + func.name if not func.name.startswith("__") else func.name))
                        funcs += f"{self.function_signature(func, extern_c=False)};\n"
            classes += "};\n"
        funcs += '\n'.join(f"{self.function_signature(f, extern_c=False)};" for f in expr.module.funcs)

        global_variables = ''.join(['extern '+self._print(d) for d in expr.module.declarations if not d.variable.is_private])

        # Print imports last to be sure that all additional_imports have been collected
        imports = [*expr.module.imports, *self._additional_imports.values()]
        imports = ''.join(self._print(i) for i in imports)

        self._in_header = False
        self.exit_scope()
        return (f"#ifndef {name.upper()}_H\n \
                #define {name.upper()}_H\n\n \
                {imports}\n \
                {global_variables}\n \
                {classes}\n \
                {funcs}\n \
                #endif // {name}_H\n")

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
