# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
Module describing the base code-wrapping class : Wrapper.
"""

from pyccel.parser.scope      import Scope
from pyccel.errors.errors     import Errors, ErrorsMode, PyccelError
from pyccel.errors.messages   import PYCCEL_RESTRICTION_TODO, PYCCEL_INTERNAL_ERROR

__all__ = ["Wrapper"]

errors = Errors()


class Wrapper:
    """
    The base class for code-wrapping subclasses.

    The base class for any classes designed to create a wrapper around code.
    Such wrappers are necessary to create an interface between two different
    languages.

    Parameters
    ----------
    verbose : int
        The level of verbosity.
    """
    start_language = None
    target_language = None

    def __init__(self, verbose):
        self._scope = None
        self._current_ast_node = None
        self._verbose = verbose

    @property
    def scope(self):
        """
        Get the current scope.

        Get the scope for the current context.

        See Also
        --------
        pyccel.parser.scope.Scope
            The type of the returned object.
        """
        return self._scope

    @scope.setter
    def scope(self, scope):
        assert isinstance(scope, Scope)
        self._scope = scope

    def exit_scope(self):
        """
        Exit the current scope and return to the enclosing scope.

        Exit the current scope and set the scope back to the value
        of the enclosing scope.
        """
        self._scope = self._scope.parent_scope

    def wrap(self, expr):
        """
        Get the wrapped version of the AST object.

        Return the AST object which allows the object `expr` printed
        in the start language to be accessed from the target language.

        Parameters
        ----------
        expr : pyccel.ast.basic.PyccelAstNode
            The expression that should be wrapped.

        Returns
        -------
        pyccel.ast.basic.PyccelAstNode
            The AST which describes the object that lets you
            access the expression.
        """
        return self._wrap(expr)

    def _wrap(self, expr):
        """
        Get the wrapped version of the AST object.

        Private function returning the AST object which is used to access
        the object `expr` from the target language.

        Parameters
        ----------
        expr : pyccel.ast.basic.PyccelAstNode
            The expression that should be wrapped.

        Returns
        -------
        pyccel.ast.basic.PyccelAstNode
            The AST which describes the object that lets you
            access the expression.
        """
        current_ast = self._current_ast_node
        if getattr(expr,'python_ast', None) is not None:
            self._current_ast_node = expr.python_ast

        classes = type(expr).mro()
        for cls in classes:
            wrap_method = '_wrap_' + cls.__name__
            if hasattr(self, wrap_method):
                if self._verbose > 2:
                    print(f">>>> Calling {type(self).__name__}.{wrap_method}")
                try:
                    obj = getattr(self, wrap_method)(expr)
                except PyccelError as err:
                    raise err
                except NotImplementedError as error:
                    errors.report(f'{error}\n'+PYCCEL_RESTRICTION_TODO,
                        symbol = self._current_ast_node, severity='fatal',
                        traceback=error.__traceback__)
                except Exception as err: #pylint: disable=broad-exception-caught
                    if ErrorsMode().value == 'user':
                        errors.report(PYCCEL_INTERNAL_ERROR,
                                symbol = self._current_ast_node, severity='fatal')
                    else:
                        raise err
                self._current_ast_node = current_ast
                return obj

        return self._wrap_not_supported(expr)

    def _wrap_not_supported(self, expr):
        """ Print an error message if the wrap function for the type
        is not implemented """
        msg = f'_wrap_{type(expr).__name__} is not yet implemented for wrapper : {type(self)}\n'
        errors.report(msg+PYCCEL_RESTRICTION_TODO, symbol = expr,
                severity='fatal')

