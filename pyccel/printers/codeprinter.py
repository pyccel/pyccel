# coding: utf-8

from __future__ import print_function, division

from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.symbol import Symbol
from sympy.core import Add, Mul, Pow, S
from sympy.core.compatibility import default_sort_key, string_types
from sympy.core.sympify import _sympify
from sympy.core.mul import _keep_coeff
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import precedence

from pyccel.types.ast import Assign
from pyccel.types.ast import FunctionDef
from pyccel.types.ast import FunctionCall
from pyccel.types.ast import ZerosLike

from pyccel.parallel.mpi import MPI
from pyccel.parallel.mpi import MPI_comm_world, MPI_comm_size

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

    def _print_Assign(self, expr):
        lhs_code = self._print(expr.lhs)
        is_procedure = False
        if isinstance(expr.rhs, FunctionDef):
            rhs_code = self._print(expr.rhs.name)
            is_procedure = (expr.rhs.kind == 'procedure')
        elif isinstance(expr.rhs, FunctionCall):
            func = expr.rhs.func
            rhs_code = self._print(func.name)
            is_procedure = (func.kind == 'procedure')
        else:
            rhs_code = self._print(expr.rhs)

        code = ''
        if (expr.status == 'unallocated') and not (expr.like is None):
            stmt = ZerosLike(lhs_code, expr.like)
            code += self._print(stmt)
            code += '\n'
        if not is_procedure:
            code += '{0} = {1}'.format(lhs_code, rhs_code)
        else:
            code_args = ''
            if (not func.arguments is None) and (len(func.arguments) > 0):
                code_args = ', '.join(self._print(i) for i in func.arguments)
                code_args = '{0},{1}'.format(code_args, lhs_code)
            else:
                code_args = lhs_code
            code = 'call {0}({1})'.format(rhs_code, code_args)
        return self._get_statement(code)

    def _print_MPI_comm_world(self, expr):
        return 'MPI_comm_world'

    def _print_MPI_comm_size(self, expr):
#        'call mpi_comm_size ({0}, size, ierr)'.format(mpi_comm_world)
        return 'MPI_comm_size'

    def _print_MPI_Assign(self, expr):
        lhs_code = self._print(expr.lhs)
        is_procedure = False
        if isinstance(expr.rhs, MPI_comm_world):
            rhs_code = self._print(expr.rhs)
            code = '{0} = {1}'.format(lhs_code, rhs_code)
        elif isinstance(expr.rhs, MPI_comm_size):
            rhs_code = self._print(expr.rhs)
            comm = self._print(expr.rhs.comm)
            size = self._print(expr.lhs)
            ierr = self._print(expr.rhs.ierr)
            code = 'call mpi_comm_size ({0}, {1}, {2})'.format(comm, size, ierr)
        return code

    def _print_Function(self, expr):
        if expr.func.__name__ in self.known_functions:
            cond_func = self.known_functions[expr.func.__name__]
            func = None
            if isinstance(cond_func, str):
                func = cond_func
            else:
                for cond, func in cond_func:
                    if cond(*expr.args):
                        break
            if func is not None:
                return "%s(%s)" % (func, self.stringify(expr.args, ", "))
        elif hasattr(expr, '_imp_') and isinstance(expr._imp_, Lambda):
            # inlined function
            return self._print(expr._imp_(*expr.args))
        else:
            name      = expr.func.__name__
            code_args = self.stringify(expr.args, ", ")
            code = '{0}({1})'.format(name, code_args)
            return code

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
        for item in args:
            if item.is_commutative and item.is_Pow and item.exp.is_Rational and item.exp.is_negative:
                if item.exp != -1:
                    b.append(Pow(item.base, -item.exp, evaluate=False))
                else:
                    b.append(Pow(item.base, -item.exp))
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
