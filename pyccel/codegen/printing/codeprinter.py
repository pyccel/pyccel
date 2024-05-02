# coding: utf-8



from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.symbol import Symbol
from sympy.core import Add, Mul, Pow, S
from sympy.core.compatibility import default_sort_key, string_types
from sympy.core.sympify import _sympify
from sympy.core.mul import _keep_coeff
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import precedence

from pyccel.ast.core import Assign,DottedVariable
from pyccel.ast.core import FunctionDef
from pyccel.ast import Real


# TODO: add examples

__all__ = ["CodePrinter"]

class CodePrinter(StrPrinter):
    """
    The base class for code-printing subclasses.
    """

    _operators = {
        'and': '&&',
        'or': '||',
        'not': '!',
    }

    def doprint(self, expr, assign_to=None):
        """
        Print the expression as code.

        expr : Expression
            The expression to be printed.

        assign_to : Symbol, MatrixSymbol, or string (optional)
            If provided, the printed code will set the expression to a
            variable with name ``assign_to``.
        """

        if isinstance(assign_to, string_types):
            assign_to = Symbol(assign_to)
        elif not isinstance(assign_to, (Basic, type(None))):
            raise TypeError("{0} cannot assign to object of type {1}".format(
                    type(self).__name__, type(assign_to)))

        if assign_to:
            expr = Assign(assign_to, expr)
        else:
            expr = _sympify(expr)

        # Do the actual printing
        lines = self._print(expr).splitlines()

        # Format the output
        return "\n".join(self._format_code(lines))



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
        return str(expr)

    def _print_Dummy(self, expr):
        # dummies must be printed as unique symbols
        return "%s_%i" % (expr.name, expr.dummy_index)  # Dummy


    def _print_And(self, expr):
        PREC = precedence(expr)
        return (" %s " % self._operators['and']).join(self.parenthesize(a, PREC)
                for a in sorted(expr.args, key=default_sort_key))

    def _print_Or(self, expr):
        PREC = precedence(expr)
        return (" %s " % self._operators['or']).join(self.parenthesize(a, PREC)
                for a in sorted(expr.args, key=default_sort_key))

    def _print_Xor(self, expr):
        if self._operators.get('xor') is None:
            return self._print_not_supported(expr)
        PREC = precedence(expr)
        return (" %s " % self._operators['xor']).join(self.parenthesize(a, PREC)
                for a in expr.args)

    def _print_Equivalent(self, expr):
        if self._operators.get('equivalent') is None:
            return self._print_not_supported(expr)
        PREC = precedence(expr)
        return (" %s " % self._operators['equivalent']).join(self.parenthesize(a, PREC)
                for a in expr.args)

    def _print_Not(self, expr):
        PREC = precedence(expr)
        return self._operators['not'] + self.parenthesize(expr.args[0], PREC)

    def _print_Mul(self, expr):
        prec = precedence(expr)

        c, e = expr.as_coeff_Mul()
        if c < 0:
            expr = _keep_coeff(-c, e)
            sign = "-"
        else:
            sign = ""

        a = []  # items in the numerator
        b = []  # items that are in the denominator (if any)

        if self.order not in ('old', 'none'):
            args = expr.as_ordered_factors()
        else:
            # use make_args in case expr was something like -x -> x
            args = Mul.make_args(expr)

        # Gather args for numerator/denominator
        flag = True
        for item in args:
            if item.is_commutative and item.is_Pow and item.exp.is_Rational and item.exp.is_negative:
                base = item.base
                if base.is_integer and flag:
                    flag = False
                    base = Real(item.base)
                    #we only need to do it once
                    #to one of the denominator args
                if item.exp != -1:
                    b.append(Pow(base, -item.exp, evaluate=False))
                else:
                    b.append(Pow(base, -item.exp))
            else:
                a.append(item)

        a = a or [S.One]

        a_str = [self.parenthesize(x, prec) for x in a]
        b_str = [self.parenthesize(x, prec) for x in b]

        if len(b) == 0:
            return sign + '*'.join(a_str)
        elif len(b) == 1:
            if len(a) == 1 and not (a[0].is_Atom or a[0].is_Add):
                return sign + "%s/" % a_str[0] + '*'.join(b_str)
            else:
                return sign + '*'.join(a_str) + "/%s" % b_str[0]
        else:
            return sign + '*'.join(a_str) + "/(%s)" % '*'.join(b_str)


    def _print_not_supported(self, expr):
        raise TypeError("{0} not supported in {1}".format(type(expr), self.language))

    # Number constants
    _print_Catalan = _print_NumberSymbol
    _print_EulerGamma = _print_NumberSymbol
    _print_GoldenRatio = _print_NumberSymbol
    _print_Exp1 = _print_NumberSymbol
    _print_Pi = _print_NumberSymbol

    # The following can not be simply translated into C or Fortran
    _print_Basic = _print_not_supported
    _print_ComplexInfinity = _print_not_supported
    _print_Derivative = _print_not_supported
    _print_dict = _print_not_supported
    _print_ExprCondPair = _print_not_supported
    _print_GeometryEntity = _print_not_supported
    _print_Infinity = _print_not_supported
    _print_Integral = _print_not_supported
    _print_Interval = _print_not_supported
    _print_Limit = _print_not_supported
    _print_list = _print_not_supported
    _print_Matrix = _print_not_supported
    _print_ImmutableMatrix = _print_not_supported
    _print_MutableDenseMatrix = _print_not_supported
    _print_MatrixBase = _print_not_supported
    _print_DeferredVector = _print_not_supported
    _print_NaN = _print_not_supported
    _print_NegativeInfinity = _print_not_supported
    _print_Normal = _print_not_supported
    _print_Order = _print_not_supported
    _print_PDF = _print_not_supported
    _print_RootOf = _print_not_supported
    _print_RootsOf = _print_not_supported
    _print_RootSum = _print_not_supported
    _print_Sample = _print_not_supported
    _print_SparseMatrix = _print_not_supported
    _print_tuple = _print_not_supported
    _print_Uniform = _print_not_supported
    _print_Unit = _print_not_supported
    _print_Wild = _print_not_supported
    _print_WildFunction = _print_not_supported
