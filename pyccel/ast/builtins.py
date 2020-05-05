# coding: utf-8
"""
The Python interpreter has a number of built-in functions and types that are
always available.

In this module we implement some of them in alphabetical order.

"""

from sympy import Symbol, Function, Tuple
from sympy import Float
from sympy import sympify
from sympy.core.assumptions import StdFactKB
from sympy.tensor import Indexed, IndexedBase
from sympy.utilities.iterables          import iterable

from .basic import Basic
from .datatypes import default_precision
from .numbers import Integer

__all__ = (
    'Bool',
    'Complex',
    'Enumerate',
    'PythonFloat',
    'Int',
    'PythonTuple',
    'Len',
    'List',
    'Map',
    'Print',
    'Range',
    'Zip',
    'python_builtin_datatype'
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
class Bool(Function):
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
class Complex(Function):
    """ Represents a call to Python's native complex() function.
    """
    is_zero = False

    def __new__(cls, arg0, arg1=Float(0)):
        return Basic.__new__(cls, arg0, arg1)

    def __init__(self, arg0, arg1=Float(0)):
        assumptions = {'complex': True}
        ass_copy = assumptions.copy()
        self._assumptions = StdFactKB(assumptions)
        self._assumptions._generator = ass_copy

    @property
    def real_part(self):
        return self._args[0]

    @property
    def imag_part(self):
        return self._args[1]

    @property
    def dtype(self):
        return 'complex'

    @property
    def shape(self):
        return None

    @property
    def rank(self):
        return 0

    @property
    def precision(self):
        return default_precision['complex']

    def __str__(self):
        return self.fprint(str)

    def _sympystr(self, printer):
        return self.fprint(str)

    def fprint(self, printer):
        """Fortran print."""
        real = printer(self.real_part)
        imag = printer(self.imag_part)
        prec = printer(self.precision)
        code = 'cmplx({0}, {1}, {2})'.format(real, imag, prec)
        return code

#==============================================================================
class Enumerate(Basic):

    """
    Represents the enumerate stmt

    """

    def __new__(cls, arg):
        if not isinstance(arg, (Symbol, Indexed, IndexedBase)):
            raise TypeError('Expecting an arg of valid type')
        return Basic.__new__(cls, arg)

    @property
    def element(self):
        return self._args[0]

#==============================================================================
class PythonFloat(Function):
    """ Represents a call to Python's native float() function.
    """
    is_zero = False
    def __new__(cls, arg):
        return Basic.__new__(cls, arg)

    def __init__(self, arg):
        assumptions = {'real': True}
        ass_copy = assumptions.copy()
        self._assumptions = StdFactKB(assumptions)
        self._assumptions._generator = ass_copy

    @property
    def arg(self):
        return self._args[0]

    @property
    def dtype(self):
        return 'float'

    @property
    def shape(self):
        return None

    @property
    def rank(self):
        return 0

    @property
    def precision(self):
        return default_precision['real']

    def __str__(self):
        return 'Float({0})'.format(str(self.arg))

    def _sympystr(self, printer):
        return self.__str__()

    def fprint(self, printer):
        """Fortran print."""
        value = printer(self.arg)
        prec  = printer(self.precision)
        code = 'Real({0}, {1})'.format(value, prec)
        return code

#==============================================================================
class Int(Function):
    """ Represents a call to Python's native int() function.
    """
    is_zero = False

    def __new__(cls, arg):
        return Basic.__new__(cls, arg)

    def __init__(self, arg):
        assumptions = {'integer': True}
        ass_copy = assumptions.copy()
        self._assumptions = StdFactKB(assumptions)
        self._assumptions._generator = ass_copy

    @property
    def arg(self):
        return self._args[0]

    @property
    def dtype(self):
        return 'int'

    @property
    def shape(self):
        return None

    @property
    def rank(self):
        return 0

    @property
    def precision(self):
        return default_precision['int']

    def fprint(self, printer):
        """Fortran print."""
        value = printer(self.arg)
        prec  = printer(self.precision)
        code  = 'Int({0}, {1})'.format(value, prec)
        return code

#==============================================================================
class PythonTuple(Function):
    """ Represents a call to Python's native tuple() function.
    """
    _iterable = True
    _arg_dtypes = None
    _is_homogeneous = False
    is_zero = False
    def __new__(cls, args):
        if not iterable(args):
            args = [args]
        args = tuple(args)

        return Basic.__new__(cls, *args)

    @property
    def dtype(self):
        return 'tuple'

    @property
    def homogeneous_dtype(self):
        assert(self.is_homogeneous)
        return self._homogeneous_dtype

    @property
    def shape(self):
        if(self._arg_dtypes is None):
            return [Integer(len(self._args))]
        else:
            shape = [Integer(len(self._args))]
            if self.is_homogeneous and self._arg_dtypes[0]['rank'] > 0:
                shape = shape + list(self._arg_dtypes[0]['shape'])

            return tuple(shape)

    @property
    def rank(self):
        return max(getattr(a,'rank',0) for a in self._args)+1

    def __getitem__(self,i):
        return self._args[i]

    def __add__(self,other):
        return PythonTuple(self._args+other._args)

    def __iter__(self):
        return self._args.__iter__()

    def __len__(self):
        return len(self._args)

    @property
    def is_homogeneous(self):
        if (self._arg_dtypes is None):
            raise RuntimeError("This function cannot be used until the type has been infered")
        return self._is_homogeneous

    def set_arg_types(self,d_vars):
        """ set the types of each argument by providing
        the list of d_vars calculated using the function
        _infere_type in parser/semantics.py

        This allows the homogeneity properties to be calculated
        """
        self._arg_dtypes = d_vars
        dtypes = [str(a['datatype']) for a in d_vars]

        #If all arguments are provided then the homogeneity must be checked
        if (len(self._args)==len(d_vars)):
            self._is_homogeneous = len(set(dtypes))==1

            if self._is_homogeneous and d_vars[0]['datatype'] == 'tuple':
                self._is_homogeneous = all(a.is_homogeneous for a in self._vars)
                if self._is_homogeneous:
                    dtypes = [str(v.homogeneous_dtype) for v in self._vars]
                    self._is_homogeneous = len(set(dtypes))==1
                    if self._is_homogeneous:
                        self._homogeneous_dtype = d_vars[0].homogeneous_dtype
            else:
                self._homogeneous_dtype = d_vars[0]['datatype']
        else:
            # If one argument is provided then the tuple must be homogeneous
            # unless it contains tuples as these tuples are not necessarily homogeneous
            assert(len(d_vars)==1)
            self._is_homogeneous = True
            self._homogeneous_dtype = d_vars[0]['datatype']
            if d_vars[0]['datatype'] == 'tuple':
                self._is_homogeneous = self._vars[0]._is_homogeneous
                if self._is_homogeneous:
                    self._is_homogeneous_dtype = self._vars[0]._homogeneous_dtype

    @property
    def arg_types(self):
        if (self._arg_dtypes is None):
            raise RuntimeError("This function cannot be used until the type has been infered")
        return self._arg_dtypes

#==============================================================================
class Len(Function):

    """
    Represents a 'len' expression in the code.
    """
    is_zero = False

    def __new__(cls, arg):
        return Basic.__new__(cls, arg)

    def __init__(self, arg):
        assumptions = {'integer': True}
        ass_copy = assumptions.copy()
        self._assumptions = StdFactKB(assumptions)
        self._assumptions._generator = ass_copy

    @property
    def arg(self):
        return self._args[0]

    @property
    def dtype(self):
        return 'int'

#==============================================================================
class List(Tuple):
    """ Represent lists in the code with dynamic memory management."""

#==============================================================================
class Map(Basic):
    """ Represents the map stmt
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

        _valid_args = (Integer, Symbol, Indexed)

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

#==============================================================================
python_builtin_datatypes_dict = {
    'bool'   : Bool,
    'float'  : PythonFloat,
    'int'    : Int,
    'complex': Complex
}

def python_builtin_datatype(name):
    """
    Given a symbol name, return the corresponding datatype.

    name: str
        Datatype as written in Python.

    """
    if not isinstance(name, str):
        raise TypeError('name must be a string')

    if name in python_builtin_datatypes_dict:
        return python_builtin_datatypes_dict[name]

    return None
