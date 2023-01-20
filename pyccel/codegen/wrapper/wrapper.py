# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

from pyccel.errors.errors     import Errors
from pyccel.errors.messages   import PYCCEL_RESTRICTION_TODO

__all__ = ["Wrapper"]

errors = Errors()


class Wrapper:
    start_language = None
    target_language = None

    @property
    def scope(self):
        """ Get the current scope
        """
        return self._scope

    def set_scope(self, scope):
        """ Change the current scope
        """
        assert scope is not None
        self._scope = scope

    def exit_scope(self):
        """ Exit the current scope and return to the enclosing scope
        """
        self._scope = self._scope.parent_scope

    def wrap(self, expr):
        return self._wrap(expr)

    def _wrap(self, expr):
        """ Return the AST object which is used to access
        the expression from the target language

        Parameters
        ----------
        expr : AST node
                The expression that should be wrapped

        Returns
        -------
        The AST which describes the object that lets you
        access the expression
        """
        classes = type(expr).mro()
        for cls in classes:
            wrap_method = '_wrap_' + cls.__name__
            if hasattr(self, wrap_method):
                return getattr(self, wrap_method)(expr)

        return self._wrap_not_supported(expr)

    def _wrap_not_supported(self, expr):
        """ Print an error message if the wrap function for the type
        is not implemented """
        msg = f'_wrap_{type(expr).__name__} is not yet implemented for wrapper : {type(self)}\n'
        errors.report(msg+PYCCEL_RESTRICTION_TODO, symbol = expr,
                severity='fatal')

