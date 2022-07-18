# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#


from pyccel.ast.basic import Basic

from pyccel.ast.core      import Assign
from pyccel.ast.internals import PyccelSymbol

from pyccel.errors.errors     import Errors
from pyccel.errors.messages   import PYCCEL_RESTRICTION_TODO

# TODO: add examples

__all__ = ["CodePrinter"]

errors = Errors()

class CodePrinter:
    """
    The base class for code-printing subclasses.
    """
    language = None
    def __init__(self):
        self._scope = None

    def doprint(self, expr, assign_to=None):
        """
        Print the expression as code.

        expr : Expression
            The expression to be printed.

        assign_to : PyccelSymbol, MatrixSymbol, or string (optional)
            If provided, the printed code will set the expression to a
            variable with name ``assign_to``.
        """

        if isinstance(assign_to, str):
            assign_to = PyccelSymbol(assign_to)
        elif not isinstance(assign_to, (Basic, type(None))):
            raise TypeError("{0} cannot assign to object of type {1}".format(
                    type(self).__name__, type(assign_to)))

        if assign_to:
            expr = Assign(assign_to, expr)

        # Do the actual printing
        lines = self._print(expr).splitlines(True)

        # Format the output
        return ''.join(self._format_code(lines))

    @property
    def scope(self):
        """ Return the scope associated with the object being printed
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

    def _print(self, expr):
        """Print the AST node in the printer language

        The printing is done by finding the appropriate function _print_X
        for the object expr. X is the type of the object expr. If this function
        does not exist then the method resolution order is used to search for
        other compatible _print_X functions. If none are found then an error is
        raised
        """

        classes = type(expr).__mro__
        for cls in classes:
            print_method = '_print_' + cls.__name__
            if hasattr(self, print_method):
                obj = getattr(self, print_method)(expr)
                return obj
        return self._print_not_supported(expr)

    def _get_statement(self, codestring):
        """Formats a codestring with the proper line ending."""
        raise NotImplementedError("This function must be implemented by "
                                  "subclass of CodePrinter.")

    def _get_comment(self, text):
        """Formats a text string as a comment."""
        raise NotImplementedError("This function must be implemented by "
                                  "subclass of CodePrinter.")

    def _declare_number_const(self, name, value):
        """Declare a numeric constant at the top of a function"""
        raise NotImplementedError("This function must be implemented by "
                                  "subclass of CodePrinter.")

    def _format_code(self, lines):
        """Take in a list of lines of code, and format them accordingly.

        This may include indenting, wrapping long lines, etc..."""
        raise NotImplementedError("This function must be implemented by "
                                  "subclass of CodePrinter.")

    def _print_NumberSymbol(self, expr):
        """ Print sympy symbols used for constants"""
        return str(expr)

    def _print_str(self, expr):
        """ Basic print functionality for strings """
        return expr

    def _print_not_supported(self, expr):
        """ Print an error message if the print function for the type
        is not implemented """
        msg = '_print_{} is not yet implemented for language : {}\n'.format(type(expr).__name__, self.language)
        errors.report(msg+PYCCEL_RESTRICTION_TODO, symbol = expr,
                severity='fatal')

    # Number constants
    _print_Catalan = _print_NumberSymbol
    _print_EulerGamma = _print_NumberSymbol
    _print_GoldenRatio = _print_NumberSymbol
    _print_Exp1 = _print_NumberSymbol
    _print_Pi = _print_NumberSymbol
