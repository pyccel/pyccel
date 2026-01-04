#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
Module representing objects (functions/variables etc) required for the interface
between Python code and C++ code (using pybind11).
This file contains classes but also many FunctionDef/Variable instances representing
objects defined by pybind11.
"""
from .basic import PyccelAstNode

__all__ = ('FunctionDeclaration',)

class FunctionDeclaration(PyccelAstNode):
    """
    A class describing how a function is added to a pybind11 module.

    A class describing how a function is added to a pybind11 module.

    Parameters
    ----------
    func : FunctionDef
        The wrapper function being declared in the scope.
    mod_var : Variable
        The variable describing the module.
    orig_func : FunctionDef
        The function being wrapped.
    """
    _my_attribute_nodes = ('_func', '_mod_var', '_orig_func')
    def __init__(self, func, mod_var, orig_func):
        self._func = func
        self._mod_var = mod_var
        self._orig_func = orig_func
        super().__init__()

    @property
    def mod_var(self):
        """
        The variable describing the module.

        The variable describing the module.
        """
        return self._mod_var

    @property
    def function(self):
        """
        The wrapper function being declared in the scope.

        The wrapper function being declared in the scope.
        """
        return self._func

    @property
    def orig_function(self):
        """
        The function being wrapped.

        The function being wrapped.
        """
        return self._orig_func
