# coding: utf-8
"""
The Python interpreter has a number of built-in functions and types that are
always available.

In this module we implement some of them in alphabetical order.

"""

from sympy import Symbol, Function, Tuple
from sympy import Expr, Not
from sympy import sympify
from sympy.tensor import Indexed, IndexedBase

from .basic     import Basic, PyccelAstNode
from .datatypes import (NativeInteger, NativeBool, NativeReal,
                        NativeComplex, NativeString, str_dtype,
                        NativeGeneric, default_precision)
from .numbers   import Integer, Float

__all__ = (
    'PythonBool',
    'PythonComplex',
    'PythonEnumerate',
    'PythonFloat',
    'PythonInt',
    'PythonTuple',
    'PythonLen',
    'PythonList',
    'PythonMap',
    'PythonPrint',
    'PythonRange',
    'PythonZip',
    'PythonMax',
    'PythonMin',
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
class PythonBool(Expr, PyccelAstNode):
    """ Represents a call to Python's native bool() function.
    """
    _rank = 0
    _shape = ()
    _precision = default_precision['bool']
    _dtype = NativeBool()

    def __new__(cls, arg):
        return Expr.__new__(cls, arg)

    @property
    def arg(self):
        return self.args[0]

    def __str__(self):
        return 'Bool({})'.format(str(self.arg))

    def _sympystr(self, printer):
        return self.__str__()

    def fprint(self, printer):
        """ Fortran printer. """
        if isinstance(self.arg.dtype, NativeBool):
            return 'logical({}, kind = {prec})'.format(printer(self.arg), prec = self.precision)
        else:
            return '{} /= 0'.format(printer(self.arg))

#==============================================================================
class PythonComplex(Expr, PyccelAstNode):
    """ Represents a call to Python's native complex() function.
    """

    _rank = 0
    _shape = ()
    _precision = default_precision['complex']
    _dtype = NativeComplex()

    def __new__(cls, arg0, arg1=Float(0)):
        return Expr.__new__(cls, arg0, arg1)

    @property
    def real_part(self):
        return self._args[0]

    @property
    def imag_part(self):
        return self._args[1]

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
class PythonEnumerate(Basic):

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
class PythonFloat(Expr, PyccelAstNode):
    """ Represents a call to Python's native float() function.
    """
    _rank = 0
    _shape = ()
    _precision = default_precision['real']
    _dtype = NativeReal()

    def __new__(cls, arg):
        return Expr.__new__(cls, arg)

    @property
    def arg(self):
        return self._args[0]


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
class PythonInt(Expr, PyccelAstNode):
    """ Represents a call to Python's native int() function.
    """

    _rank      = 0
    _shape     = ()
    _precision = default_precision['integer']
    _dtype     = NativeInteger()

    def __new__(cls, arg):
        return Expr.__new__(cls, arg)

    @property
    def arg(self):
        return self._args[0]

    def fprint(self, printer):
        """Fortran print."""
        value = printer(self.arg)
        prec  = printer(self.precision)
        if (self.arg.dtype is NativeBool()):
            code = 'MERGE(1_8, 0_8, {})'.format(value)
        else:
            code  = 'Int({0}, {1})'.format(value, prec)
        return code

#==============================================================================
class PythonTuple(Expr, PyccelAstNode):
    """ Represents a call to Python's native tuple() function.
    """
    _iterable        = True
    _is_homogeneous  = False

    def __new__(cls, *args):
        return Expr.__new__(cls, *args)

    def __init__(self, *args):
        if self.stage == 'syntactic' or len(args) == 0:
            return
        is_homogeneous = all(a.dtype is not NativeGeneric() and \
                             args[0].dtype == a.dtype and \
                             args[0].rank  == a.rank  for a in args[1:])
        self._inconsistent_shape = not all(args[0].shape==a.shape   for a in args[1:])
        self._is_homogeneous = is_homogeneous
        if is_homogeneous:
            integers  = [a for a in args if a.dtype is NativeInteger()]
            reals     = [a for a in args if a.dtype is NativeReal()]
            complexes = [a for a in args if a.dtype is NativeComplex()]
            bools     = [a for a in args if a.dtype is NativeBool()]
            strs      = [a for a in args if a.dtype is NativeString()]
            if strs:
                self._dtype = NativeString()
                self._rank  = 0
                self._shape = ()
            else:
                if complexes:
                    self._dtype     = NativeComplex()
                    self._precision = max(a.precision for a in complexes)
                elif reals:
                    self._dtype     = NativeReal()
                    self._precision = max(a.precision for a in reals)
                elif integers:
                    self._dtype     = NativeInteger()
                    self._precision = max(a.precision for a in integers)
                elif bools:
                    self._dtype     = NativeBool()
                    self._precision  = max(a.precision for a in bools)
                else:
                    raise TypeError('cannot determine the type of {}'.format(self))


                shapes = [a.shape for a in args]

                if all(sh is not None for sh in shapes):
                    self._shape = (Integer(len(args)), ) + shapes[0]
                    self._rank  = len(self._shape)
                else:
                    self._rank = max(a.rank for a in args) + 1
        else:
            self._rank      = max(a.rank for a in args) + 1
            self._dtype     = NativeGeneric()
            self._precision = 0
            self._shape     = (Integer(len(args)), ) + args[0].shape

    def __getitem__(self,i):
        return self._args[i]

    def __add__(self,other):
        return PythonTuple(*(self._args + other._args))

    def __iter__(self):
        return self._args.__iter__()

    def __len__(self):
        return len(self._args)

    @property
    def is_homogeneous(self):
        return self._is_homogeneous

    @property
    def inconsistent_shape(self):
        return self._inconsistent_shape

#==============================================================================
class PythonLen(Function, PyccelAstNode):

    """
    Represents a 'len' expression in the code.
    """

    _rank      = 0
    _shape     = ()
    _precision = default_precision['int']
    _dtype     = NativeInteger()

    def __new__(cls, arg):
        return Basic.__new__(cls, arg)

    @property
    def arg(self):
        return self._args[0]

#==============================================================================
class PythonList(Tuple, PyccelAstNode):
    """ Represent lists in the code with dynamic memory management."""
    def __init__(self, *args, **kwargs):
        if self.stage == 'syntactic':
            return
        bools     = [a for a in args if a.dtype is NativeBool()]
        integers  = [a for a in args if a.dtype is NativeInteger()]
        reals     = [a for a in args if a.dtype is NativeReal()]
        complexes = [a for a in args if a.dtype is NativeComplex()]
        strs      = [a for a in args if a.dtype is NativeString()]
        if strs:
            self._dtype = NativeString()
            self._rank  = 0
            self._shape = ()
            assert len(integers + reals + complexes) == 0
        else:
            if complexes:
                self._dtype     = NativeComplex()
                self._precision = max(a.precision for a in complexes)
            elif reals:
                self._dtype     = NativeReal()
                self._precision = max(a.precision for a in reals)
            elif integers:
                self._dtype     = NativeInteger()
                self._precision = max(a.precision for a in integers)
            elif bools:
                self._dtype     = NativeBool()
                self._precision  = max(a.precision for a in bools)
            else:
                raise TypeError('cannot determine the type of {}'.format(self))

            shapes = [a.shape for a in args]

            if all(sh is not None for sh in shapes):
                assert all(sh==shapes[0] for sh in shapes)
                self._shape = (Integer(len(args)), ) + shapes[0]
                self._rank  = len(self._shape)
            else:
                self._rank = max(a.rank for a in args) + 1
#==============================================================================
class PythonMap(Basic):
    """ Represents the map stmt
    """
    def __new__(cls, *args):
        if len(args)<2:
            raise TypeError('wrong number of arguments')
        return Basic.__new__(cls, *args)

#==============================================================================
class PythonPrint(Basic):

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
class PythonRange(Basic):

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
        start = Integer(0)
        stop = None
        step = Integer(1)

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
class PythonZip(Basic):

    """
    Represents a zip stmt.

    """

    def __new__(cls, *args):
        if not isinstance(args, (tuple, list, Tuple)):
            raise TypeError('args must be an iterable')
        elif len(args) < 2:
            raise ValueError('args must be of length > 2')
        return Basic.__new__(cls, *args)

    @property
    def element(self):
        return self._args[0]

#==============================================================================
class PythonAbs(Function, PyccelAstNode):
    """Represents a call to  python abs for code generation.

    arg : Variable
    """
    def __init__(self, x):
        self._shape     = x.shape
        self._rank      = x.rank
        self._dtype     = NativeInteger() if x.dtype is NativeInteger() else NativeReal()
        self._precision = default_precision[str_dtype(self._dtype)]
        self._order     = x.order

    @property
    def arg(self):
        return self._args[0]

#==============================================================================
class PythonSum(Function, PyccelAstNode):
    """Represents a call to  python sum for code generation.

    arg : list , tuple , PythonTuple, Tuple, List, Variable
    """

    def __new__(cls, arg):
        if not isinstance(arg, (list, tuple, PythonTuple, Tuple, PythonList,
                                Variable, Expr)): # pylint: disable=undefined-variable
            raise TypeError('Uknown type of  %s.' % type(arg))

        return Basic.__new__(cls, arg)

    def __init__(self, arg):
        self._dtype = arg.dtype
        self._rank  = 0
        self._shape = ()
        self._precision = default_precision[str_dtype(self._dtype)]

    @property
    def arg(self):
        return self._args[0]

#==============================================================================
class PythonMax(Function, PyccelAstNode):
    """Represents a call to  python max for code generation.

    arg : list , tuple , PythonTuple, Tuple, List
    """
    def __new__(cls, arg):
        if not isinstance(arg, (list, tuple, PythonTuple, Tuple, PythonList)):
            raise TypeError('Uknown type of  %s.' % type(arg))
        return Basic.__new__(cls, arg)

    def __init__(self, x):
        self._shape     = ()
        self._rank      = 0
        self._dtype     = x.dtype
        self._precision = x.precision


#==============================================================================
class PythonMin(Function, PyccelAstNode):
    """Represents a call to  python min for code generation.

    arg : list , tuple , PythonTuple, Tuple, List, Variable
    """
    def __init__(self, x):
        self._shape     = ()
        self._rank      = 0
        self._dtype     = x.dtype
        self._precision = x.precision

#==============================================================================
python_builtin_datatypes_dict = {
    'bool'   : PythonBool,
    'float'  : PythonFloat,
    'int'    : PythonInt,
    'complex': PythonComplex
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

builtin_functions_dict = {
    'abs'      : PythonAbs,
    'range'    : PythonRange,
    'zip'      : PythonZip,
    'enumerate': PythonEnumerate,
    'int'      : PythonInt,
    'float'    : PythonFloat,
    'complex'  : PythonComplex,
    'bool'     : PythonBool,
    'sum'      : PythonSum,
    'len'      : PythonLen,
    'max'      : PythonMax,
    'min'      : PythonMin,
    'not'      : Not,   # TODO [YG, 20.05.2020]: do not use Sympy's Not
}
