# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#


from pyccel.ast.basic import PyccelAstNode

from pyccel.ast.core      import Module, ModuleHeader, Program
from pyccel.ast.internals import PyccelSymbol

from pyccel.errors.errors     import Errors
from pyccel.errors.messages   import PYCCEL_RESTRICTION_TODO

# TODO: add examples

__all__ = ["CodePrinter"]

errors = Errors()

class CodePrinter:
    """
    The base class for code-printing subclasses.

    The base class from which code printers inherit. The sub-classes should define a language
    and `_print_X` functions.
    """
    language = None
    def __init__(self):
        self._scope = None
        self._additional_imports = {}

    def doprint(self, expr):
        """
        Print the expression as code.

        Print the expression as code.

        Parameters
        ----------
        expr : Expression
            The expression to be printed.

        Returns
        -------
        str
            The generated code.
        """
        assert isinstance(expr, (Module, ModuleHeader, Program))

        # Do the actual printing
        lines = self._print(expr).splitlines(True)

        # Format the output
        return ''.join(self._format_code(lines))

    def get_additional_imports(self):
        """
        Get any additional imports collected during the printing stage.

        Get any additional imports collected during the printing stage.
        This is necessary to correctly compile the files.

        Returns
        -------
        dict[str, Import]
            A dictionary mapping the include strings to the import module.
        """
        return self._additional_imports

    def add_import(self, import_obj):
        """
        Add a new import to the current context.

        Add a new import to the current context. This allows the import to be recognised
        at the compiling/linking stage. If the source of the import is not new then any
        new targets are added to the Import object.

        Parameters
        ----------
        import_obj : Import
            The AST node describing the import.
        """
        if import_obj.source not in self._additional_imports:
            self._additional_imports[import_obj.source] = import_obj
        elif import_obj.target:
            self._additional_imports[import_obj.source].define_target(import_obj.target)

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
