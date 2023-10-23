# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

from pyccel.codegen.printing.ccode import CCodePrinter

from pyccel.ast.bind_c     import BindCPointer
from pyccel.ast.bind_c     import BindCModule, BindCFunctionDef
from pyccel.ast.core       import FunctionAddress, SeparatorComment
from pyccel.ast.core       import Import, Module
from pyccel.ast.cwrapper   import PyBuildValueNode
from pyccel.ast.cwrapper   import Py_None
from pyccel.ast.cwrapper   import PyccelPyObject
from pyccel.ast.literals   import LiteralString, Nil
from pyccel.ast.c_concepts import ObjectAddress

from pyccel.errors.errors  import Errors

__all__ = ("CWrapperCodePrinter", "cwrappercode")

errors = Errors()

module_imports = [Import('numpy_version', Module('numpy_version',(),())),
            Import('numpy/arrayobject', Module('numpy/arrayobject',(),())),
            Import('cwrapper', Module('cwrapper',(),()))]

cwrapper_ndarray_import = Import('cwrapper_ndarrays', Module('cwrapper_ndarrays', (), ()))


class CWrapperCodePrinter(CCodePrinter):
    """
    A printer for printing the C-Python interface.

    A printer to convert Pyccel's AST describing a translated module,
    to strings of C code which provide an interface between the module
    and Python code.
    As for all printers the navigation of this file is done via _print_X
    functions.

    Parameters
    ----------
    filename : str
            The name of the file being pyccelised.
    target_language : str
            The language which the code was translated to [fortran/c].
    **settings : dict
            Any additional arguments which are necessary for CCodePrinter.
    """
    dtype_registry = {**CCodePrinter.dtype_registry,
                      (PyccelPyObject() , 0) : 'PyObject',
                      (BindCPointer()   , 0) : 'void'}

    def __init__(self, filename, target_language, **settings):
        CCodePrinter.__init__(self, filename, **settings)
        self._target_language = target_language
        self._to_free_PyObject_list = []
        self._function_wrapper_names = dict()
        self._module_name = None

    # --------------------------------------------------------------------
    #                       Helper functions
    # --------------------------------------------------------------------

    def is_c_pointer(self, a):
        """
        Indicate whether the object is a pointer in C code.

        This function extends `CCodePrinter.is_c_pointer` to specify more objects
        which are always accesed via a C pointer.

        Parameters
        ----------
        a : TypedAstNode
            The object whose storage we are enquiring about.

        Returns
        -------
        bool
            True if a C pointer, False otherwise.

        See Also
        --------
        CCodePrinter.is_c_pointer : The extended function.
        """
        if isinstance(a.dtype, PyccelPyObject):
            return True
        elif isinstance(a, PyBuildValueNode):
            return True
        else:
            return CCodePrinter.is_c_pointer(self,a)

    def get_python_name(self, scope, obj):
        """
        Get the name of object as defined in the original python code.

        Get the name of the object as it was originally defined in the
        Python code being translated. This name may have changed before
        the printing stage in the case of name clashes or language interfaces.

        Parameters
        ----------
        scope : pyccel.parser.scope.Scope
            The scope where the object was defined.

        obj : pyccel.ast.basic.PyccelAstNode
            The object whose name we wish to identify.

        Returns
        -------
        str
            The original name of the object.
        """
        if isinstance(obj, BindCFunctionDef):
            return scope.get_python_name(obj.original_function.name)
        elif isinstance(obj, BindCModule):
            return obj.original_module.name
        else:
            return scope.get_python_name(obj.name)

    def function_signature(self, expr, print_arg_names = True):
        args = list(expr.arguments)
        if any([isinstance(a.var, FunctionAddress) for a in args]):
            # Functions with function addresses as arguments cannot be
            # exposed to python so there is no need to print their signature
            return ''
        else:
            return CCodePrinter.function_signature(self, expr, print_arg_names)

    def get_declare_type(self, expr):
        """
        Get the string which describes the type in a declaration.

        This function extends `CCodePrinter.get_declare_type` to specify types
        which are only relevant in the C-Python interface.

        Parameters
        ----------
        expr : Variable
            The variable whose type should be described.

        Returns
        -------
        str
            The code describing the type.

        Raises
        ------
        PyccelCodegenError
            If the type is not supported in the C code or the rank is too large.

        See Also
        --------
        CCodePrinter.get_declare_type : The extended function.
        """
        if expr.dtype is BindCPointer():
            return 'void*'
        return CCodePrinter.get_declare_type(self, expr)

    def _handle_is_operator(self, Op, expr):
        """
        Get the code to print an `is` or `is not` expression.

        Get the code to print an `is` or `is not` expression. These two operators
        function similarly so this helper function reduces code duplication.
        This function overrides CCodePrinter._handle_is_operator to add the
        handling of `Py_None`.

        Parameters
        ----------
        Op : str
            The C operator representing "is" or "is not".

        expr : PyccelIs/PyccelIsNot
            The expression being printed.

        Returns
        -------
        str
            The code describing the expression.

        Raises
        ------
        PyccelError : Raised if the comparison is poorly defined.
        """
        if expr.rhs is Py_None:
            lhs = ObjectAddress(expr.lhs)
            rhs = ObjectAddress(expr.rhs)
            lhs = self._print(lhs)
            rhs = self._print(rhs)
            return f'{lhs} {Op} {rhs}'
        else:
            return super()._handle_is_operator(Op, expr)

    #--------------------------------------------------------------------
    #                 _print_ClassName functions
    #--------------------------------------------------------------------

    def _print_DottedName(self, expr):
        names = expr.name
        return '.'.join(self._print(n) for n in names)

    def _print_PyInterface(self, expr):
        funcs_to_print = (*expr.functions, expr.type_check_func, expr.interface_func)
        return '\n'.join(self._print(f) for f in funcs_to_print)

    def _print_PyArg_ParseTupleNode(self, expr):
        name    = 'PyArg_ParseTupleAndKeywords'
        pyarg   = expr.pyarg
        pykwarg = expr.pykwarg
        flags   = expr.flags
        # All args are modified so even pointers are passed by address
        args    = ', '.join(f'&{a.name}' for a in expr.args)

        if expr.args:
            code = f'{name}({pyarg}, {pykwarg}, "{flags}", {expr.arg_names.name}, {args})'
        else :
            code =f'{name}({pyarg}, {pykwarg}, "", {expr.arg_names.name})'

        return code

    def _print_PyBuildValueNode(self, expr):
        name  = 'Py_BuildValue'
        flags = expr.flags
        args  = ', '.join(self._print(a) for a in expr.args)
        #to change for args rank 1 +
        if expr.args:
            code = f'(*{name}("{flags}", {args}))'
        else :
            code = f'(*{name}(""))'
        return code

    def _print_PyArgKeywords(self, expr):
        arg_names = ',\n'.join([f'"{a}"' for a in expr.arg_names] + [self._print(Nil())])
        return (f'static char *{expr.name}[] = {{\n'
                        f'{arg_names}\n'
                        '};\n')

    def _print_PyModule_AddObject(self, expr):
        name = self._print(expr.name)
        var  = self._print(expr.variable)
        return f'PyModule_AddObject({expr.mod_name}, {name}, {var})'

    def _print_PyModule(self, expr):
        scope = expr.scope
        self.set_scope(scope)
        # The initialisation and deallocation shouldn't be exposed to python
        funcs_to_wrap = [f for f in expr.funcs if f not in (expr.init_func, expr.free_func)]

        # Insert declared objects into scope
        variables = expr.original_module.variables if isinstance(expr, BindCModule) else expr.variables
        for f in expr.funcs:
            scope.insert_symbol(f.name.lower())
        for v in variables:
            if not v.is_private:
                scope.insert_symbol(v.name.lower())

        funcs = []

        self._module_name  = self.get_python_name(scope, expr)
        sep = self._print(SeparatorComment(40))

        function_signatures = ''.join(self.function_signature(f, print_arg_names = False) + ';\n' for f in expr.external_funcs)

        interface_funcs = [f.name for i in expr.interfaces for f in i.functions]
        funcs += [*expr.interfaces, *(f for f in funcs_to_wrap if f.name not in interface_funcs)]

        self._in_header = True
        decs = ''.join(self._print(d) for d in expr.declarations)
        self._in_header = False

        function_defs = '\n'.join(self._print(f) for f in funcs)

        method_def_func = ''.join('\n'.join(('{',
                                            '"'+self.get_python_name(expr.scope, f.original_function)+'",',
                                            f'(PyCFunction){f.name},',
                                            'METH_VARARGS | METH_KEYWORDS,',
                                            self._print(LiteralString('\n'.join(f.doc_string.comments))) \
                                                       if f.doc_string else '""',
                                            '},')) for f in funcs if f is not expr.init_func and not getattr(f, 'is_header', False))

        slots_name = self.scope.get_new_name(f'{expr.name}_slots')
        exec_func_name = expr.init_func.name
        slots_def = (f'static PyModuleDef_Slot {slots_name}[] = {{\n'
                     f'{{Py_mod_exec, {exec_func_name}}},\n'
                     '{0, NULL},\n'
                     '};\n')

        method_def_name = self.scope.get_new_name(f'{expr.name}_methods')
        method_def = (f'static PyMethodDef {method_def_name}[] = {{\n'
                        f'{method_def_func}'
                        '{ NULL, NULL, 0, NULL}\n'
                        '};\n')

        module_def_name = self.scope.get_new_name(f'{expr.name}_module')
        module_def = (f'static struct PyModuleDef {module_def_name} = {{\n'
                'PyModuleDef_HEAD_INIT,\n'
                '/* name of module */\n'
                f'"{self._module_name}",\n'
                '/* module documentation, may be NULL */\n'
                'NULL,\n' #TODO: Add documentation
                '/* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */\n'
                '0,\n'
                f'{method_def_name},\n'
                f'{slots_name}\n'
                '};\n')

        exec_func = self._print(expr.init_func)

        init_func = (f'PyMODINIT_FUNC PyInit_{self._module_name}(void)\n{{\n'
                'import_array();\n'
                f'return PyModuleDef_Init(&{module_def_name});\n'
                '}\n')

        # Print imports last to be sure that all additional_imports have been collected
        for i in expr.imports:
            self.add_import(i)
        imports  = module_imports.copy()
        imports += self._additional_imports.values()
        imports  = ''.join(self._print(i) for i in imports)

        self.exit_scope()

        return '\n'.join(['#define PY_ARRAY_UNIQUE_SYMBOL CWRAPPER_ARRAY_API',
                imports, decs, function_signatures, sep, sep, function_defs,
                exec_func, sep, method_def, sep, slots_def, sep, module_def,
                sep, init_func])

def cwrappercode(expr, filename, target_language, assign_to=None, **settings):
    """Converts an expr to a string of c wrapper code

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

    return CWrapperCodePrinter(filename, target_language, **settings).doprint(expr, assign_to)
