#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
""" Functions for printing PyBind11 code.
"""
from pyccel.codegen.printing.cppcode import CppCodePrinter

from pyccel.ast.core       import SeparatorComment
from pyccel.ast.core       import Import, Module

__all__ = ("PyBindCodePrinter",)

module_imports = [Import('numpy_version', Module('numpy_version',(),())),
            Import('numpy/arrayobject', Module('numpy/arrayobject',(),())),
            Import('cwrapper', Module('cwrapper',(),()))]

pybind_imports = {'complex': Import('pybind11/complex.h', Module('pybind11/complex.h',(),()))}

class PyBindCodePrinter(CppCodePrinter):
    """
    A printer for printing the C++-Python interface.

    A printer to convert Pyccel's AST describing a translated module,
    to strings of PyBind11 code which provide an interface between the module
    and Python code.
    As for all printers the navigation of this file is done via _print_X
    functions.

    Parameters
    ----------
    filename : str
            The name of the file being pyccelised.
    **settings : dict
            Any additional arguments which are necessary for CCodePrinter.
    """

    def __init__(self, filename, **settings):
        CppCodePrinter.__init__(self, filename, **settings)
        self._to_free_PyObject_list = []
        self._function_wrapper_names = {}
        self._module_name = None
        self._current_module = None

    #-----------------------------------------------------------------------
    def _print_NumpyComplex64Type(self, expr):
        self.add_import(pybind_imports['complex'])
        return super()._print_NumpyComplex64Type(expr)

    def _print_NumpyComplex128Type(self, expr):
        self.add_import(pybind_imports['complex'])
        return super()._print_NumpyComplex64Type(expr)

    def _print_PythonNativeComplex(self, expr):
        self.add_import(pybind_imports['complex'])
        return super()._print_NumpyComplex64Type(expr)

    #-----------------------------------------------------------------------
    #                              Pybind11 methods
    #-----------------------------------------------------------------------

    def _print_PyModule(self, expr):
        scope = expr.scope
        self.set_scope(scope)
        self._current_module = expr

        self._module_name  = expr.name
        sep = self._print(SeparatorComment(40))

        init_func = self._print(expr.init_func)

        funcs = '\n'.join(self._print(f) for f in expr.funcs)

        pymod_name = f'{expr.name}_wrapper'
        imports = [Import(pymod_name, Module(pymod_name,(),())), *self._additional_imports.values()]
        imports = ''.join(self._print(i) for i in imports)
        imports = '#include <pybind11/pybind11.h>\n' + imports

        self.exit_scope()
        self._current_module = None

        return '\n'.join((f'#define {pymod_name.upper()}\n',
                          imports, sep, funcs, sep, init_func))

    def _print_PyModInitFunc(self, expr):
        my_vars = expr.scope.variables
        assert len(my_vars) == 1
        mod = next(iter(my_vars.values()))
        code = ''.join((f'PYBIND11_MODULE({expr.name}, {self._print(mod)})\n',
                        '{\n',
                        self._print(expr.body),
                        '}\n'))
        return code

    def _print_FunctionDeclaration(self, expr):
        orig_func = expr.orig_function
        orig_name = orig_func.name
        wrap_func = expr.function
        wrap_name = wrap_func.name
        args = [f'"{orig_name}"',
                f'&{wrap_name}']
        pos_only = True
        kw_only = False
        for a in wrap_func.arguments:
            if a.is_posonly != pos_only:
                args.append(f'pybind11::pos_only()')
                pos_only = a.is_posonly
            if a.is_kwonly != kw_only:
                args.append(f'pybind11::kw_only()')
                pos_only = a.is_posonly
            args.append(f'pybind11::arg("{a.name}")')
        if orig_func.docstring:
            args.append(f'pybind11::doc("{orig_func.docstring}")')
        args_str = ',\n'.join(args)
        return f'{expr.mod_var}.def({args_str});\n'

    def _print_Import(self, expr):
        if expr in pybind_imports.values():
            return f'#include <{expr.source}>\n'
        else:
            return super()._print_Import(expr)
