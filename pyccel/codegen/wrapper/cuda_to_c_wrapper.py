# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
Module describing the code-wrapping class : CudaToCWrapper
which creates an interface exposing Cuda code to C.
"""

from pyccel.codegen.wrapper.c_to_python_wrapper import CToPythonWrapper
from pyccel.parser.scope import Scope
from pyccel.ast.core import Import, FunctionDef
from pyccel.ast.cwrapper      import PyModule
from pyccel.ast.core import Module

cwrapper_ndarray_imports = [Import('cwrapper_ndarrays', Module('cwrapper_ndarrays', (), ()))]

class CudaToCWrapper(CToPythonWrapper):
    """
    Class for creating a wrapper exposing Fortran code to C.

    A class which provides all necessary functions for wrapping different AST
    objects such that the resulting AST is C-compatible. This new AST is
    printed as an intermediary layer.
    """
    def __init__(self, file_location):
        self._wrapper_names_dict = {}
        super().__init__(file_location)

    def _wrap_Module(self, expr):
        """
        Build a `PyModule` from a `Module`.

        Create a `PyModule` which wraps a C-compatible `Module`.

        Parameters
        ----------
        expr : Module
            The module which can be called from C.

        Returns
        -------
        PyModule
            The module which can be called from Python.
        """
        # Define scope
        scope = expr.scope
        mod_scope = Scope(used_symbols = scope.local_used_symbols.copy(), original_symbols = scope.python_names.copy())
        self.scope = mod_scope

        imports = [self._wrap(i) for i in getattr(expr, 'original_module', expr).imports]
        imports = [i for i in imports if i]

        # Wrap classes
        classes = [self._wrap(i) for i in expr.classes]

        # Wrap functions
        funcs_to_wrap = [f for f in expr.funcs if f not in (expr.init_func, expr.free_func)]
        funcs_to_wrap = [f for f in funcs_to_wrap if not f.is_inline]

        # Add any functions removed by the Fortran printer
        removed_functions = getattr(expr, 'removed_functions', None)
        if removed_functions:
            funcs_to_wrap.extend(removed_functions)

        funcs = [self._wrap(f) for f in funcs_to_wrap]

        # Wrap interfaces
        interfaces = [self._wrap(i) for i in expr.interfaces if not i.is_inline]

        init_func = self._build_module_init_function(expr, imports)

        API_var, import_func = self._build_module_import_function(expr)

        self.exit_scope()
        imports += cwrapper_ndarray_imports if self._wrapping_arrays else []

        original_mod = getattr(expr, 'original_module', expr)

        external_funcs = []
        # Add external functions for normal functions
        for f in expr.funcs:
            external_funcs.append(FunctionDef(f.name.lower(), f.arguments, f.results, [], is_header = True, scope = Scope()))

        for c in expr.classes:
            m = c.new_func
            external_funcs.append(FunctionDef(m.name, m.arguments, m.results, [], is_header = True, scope = Scope()))
            for m in c.methods:
                external_funcs.append(FunctionDef(m.name, m.arguments, m.results, [], is_header = True, scope = Scope()))
            for i in c.interfaces:
                for f in i.functions:
                    external_funcs.append(FunctionDef(f.name, f.arguments, f.results, [], is_header = True, scope = Scope()))
            for a in c.attributes:
                for f in (a.getter, a.setter):
                    external_funcs.append(FunctionDef(f.name, f.arguments, f.results, [], is_header = True, scope = Scope()))
        return PyModule(original_mod.name, [API_var], funcs, imports = imports,
                        interfaces = interfaces, classes = classes, scope = mod_scope,
                        init_func = init_func, import_func = import_func , external_funcs = external_funcs)

