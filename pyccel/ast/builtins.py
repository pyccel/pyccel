# coding: utf-8
"""
The Python interpreter has a number of built-in functions and types that are
always available.

In this module we implement some of them in alphabetical order.

"""

from sympy import Symbol, Function, Tuple
from sympy import Integer as sp_Integer
from sympy import sympify
from sympy.core.function import Application
from sympy.core.assumptions import StdFactKB
from sympy.tensor import Indexed, IndexedBase

from .basic import Basic

__all__ = (
    'Bool',
    'Enumerate',
    'Len',
    'List',
    'Map',
    'Print',
    'Range',
    'Zip',
)

#==============================================================================
# TODO [YG, 06.03.2020]: avoid core duplication between builtins and core
local_sympify = {
    'N'    : Symbol('N'),
    'S'    : Symbol('S'),
    'zeros': Symbol('zeros'),
    'ones' : Symbol('ones'),
    'Point': Symbol('Point')
}

#==============================================================================
class Bool(Application):
    """ Represents a call to Python's native bool() function.
    """
    is_Boolean = True

    def __new__(cls, arg):
        if arg.is_Boolean:
            return arg
        return Basic.__new__(cls, arg)

    @property
    def arg(self):
        return self.args[0]

    @property
    def dtype(self):
        return 'bool'

    @property
    def shape(self):
        return None

    @property
    def rank(self):
        return 0

    @property
    def precision(self):
        return default_precision['bool']

    def __str__(self):
        return 'Bool({})'.format(str(self.arg))

    def _sympystr(self, printer):
        return self.__str__()

    def fprint(self, printer):
        """ Fortran printer. """
        return 'merge(.true., .false., ({}) /= 0)'.format(printer(self.arg))

#==============================================================================
class Enumerate(Basic):

    """
    Reresents the enumerate stmt

    """

    def __new__(cls, arg):
        if not isinstance(arg, (Symbol, Indexed, IndexedBase)):
            raise TypeError('Expecting an arg of valid type')
        return Basic.__new__(cls, arg)

    @property
    def element(self):
        return self._args[0]

#==============================================================================
class Len(Function):

    """
    Represents a 'len' expression in the code.
    """

    def __new__(cls, arg):
        obj = Basic.__new__(cls, arg)
        assumptions = {'integer': True}
        ass_copy = assumptions.copy()
        obj._assumptions = StdFactKB(assumptions)
        obj._assumptions._generator = ass_copy
        return obj

    @property
    def arg(self):
        return self._args[0]

    @property
    def dtype(self):
        return 'int'

#==============================================================================
class List(Tuple):

    """Represent lists in the code with dynamic memory management."""

    pass

#==============================================================================
class Map(Basic):
    """
    Reresents the map stmt

    """

    def __new__(cls, *args):
        if len(args)<2:
            raise TypeError('wrong number of arguments')
        return Basic.__new__(cls, *args)

#==============================================================================
class Print(Basic):

    """Represents a print function in the code.

    expr : sympy expr
        The expression to return.

    Examples

    >>> from sympy import symbols
    >>> from pyccel.ast.core import Print
    >>> n,m = symbols('n,m')
    >>> Print(('results', n,m))
    Print((results, n, m))
    """

    def __new__(cls, expr):
        if not isinstance(expr, list):
            expr = sympify(expr, locals=local_sympify)
        return Basic.__new__(cls, expr)

    @property
    def expr(self):
        return self._args[0]

#==============================================================================
class Range(Basic):

    """
    Represents a range.

    Examples

    >>> from pyccel.ast.core import Variable
    >>> from pyccel.ast.core import Range
    >>> from sympy import Symbol
    >>> s = Variable('int', 's')
    >>> e = Symbol('e')
    >>> Range(s, e, 1)
    Range(0, n, 1)
    """

    def __new__(cls, *args):
        start = 0
        stop = None
        step = 1

        _valid_args = (sp_Integer, Symbol, Indexed)

        if isinstance(args, (tuple, list, Tuple)):
            if len(args) == 1:
                stop = args[0]
            elif len(args) == 2:
                start = args[0]
                stop = args[1]
            elif len(args) == 3:
                start = args[0]
                stop = args[1]
                step = args[2]
            else:
                raise ValueError('Range has at most 3 arguments')
        elif isinstance(args, _valid_args):
            stop = args
        else:
            raise TypeError('expecting a list or valid stop')

        return Basic.__new__(cls, start, stop, step)

    @property
    def start(self):
        return self._args[0]

    @property
    def stop(self):
        return self._args[1]

    @property
    def step(self):
        return self._args[2]

    @property
    def size(self):
        return (self.stop - self.start) / self.step


#==============================================================================
class Zip(Basic):

    """
    Represents a zip stmt.

    """

    def __new__(cls, *args):
        if not isinstance(args, (tuple, list, Tuple)):
            raise TypeError('args must be an iterable')
        elif len(args) < 2:
            raise ValueError('args must be of lenght > 2')
        return Basic.__new__(cls, *args)

    @property
    def element(self):
        return self._args[0]
