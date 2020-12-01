""" This module contains all literal types
"""
from sympy               import Expr
from sympy               import Integer as sp_Integer
from sympy               import Float as sp_Float
from sympy.logic.boolalg import BooleanTrue as sp_BooleanTrue, BooleanFalse as sp_BooleanFalse

from .basic              import PyccelAstNode, Basic
from .datatypes          import (NativeInteger, NativeBool, NativeReal,
                                  NativeComplex, NativeString, default_precision)

__all__ = (
    'LiteralTrue',
    'LiteralFalse',
    'LiteralInteger',
    'LiteralFloat',
    'LiteralComplex',
    'LiteralImaginaryUnit',
    'LiteralString',
    'get_default_literal_value'
)

#------------------------------------------------------------------------------
class Literal(PyccelAstNode):
    """
    Represents a python literal
    This class is abstract and should be implemented for each dtype
    """
    _rank      = 0
    _shape     = ()

    def __init__(self, precision):
        if not isinstance(precision, int):
            raise TypeError("precision must be an integer")
        self._precision = precision
    @PyccelAstNode.precision.setter
    def precision(self, precision):
        """ Set precision for a literal class"""
        self._precision = precision

#------------------------------------------------------------------------------
class LiteralTrue(sp_BooleanTrue, Literal):
    """Represents the python value True"""
    _dtype     = NativeBool()
    def __init__(self, precision = default_precision['bool']):
        Literal.__init__(self, precision)

#------------------------------------------------------------------------------
class LiteralFalse(sp_BooleanFalse, Literal):
    """Represents the python value False"""
    _dtype     = NativeBool()
    def __init__(self,precision = default_precision['bool']):
        Literal.__init__(self, precision)

#------------------------------------------------------------------------------
class LiteralInteger(sp_Integer, Literal):
    """Represents an integer literal in python"""
    _dtype     = NativeInteger()
    def __new__(cls, val, precision = default_precision['integer']):
        ival = int(val)
        obj = Expr.__new__(cls, ival)
        obj._precision = precision
        obj.p = ival
        return obj
    def __init__(self, value, precision = default_precision['integer']):
        Literal.__init__(self, precision)

#------------------------------------------------------------------------------
class LiteralFloat(sp_Float, Literal):
    """Represents a float literal in python"""
    _dtype     = NativeReal()
    def __init__(self, value, *, precision = default_precision['float']):
        Literal.__init__(self, precision)


#------------------------------------------------------------------------------
class LiteralComplex(Basic, Literal):
    """Represents a complex literal in python"""
    _dtype     = NativeComplex()

    def __new__(cls, real, imag):
        return Basic.__new__(cls, real, imag)

    def __init__(self, real, imag, precision = default_precision['complex']):
        Basic.__init__(self)
        self._real_part = real
        self._imag_part = imag
        Literal.__init__(self, precision)

    @property
    def real(self):
        """ Return the real part of the complex literal """
        return self._real_part

    @property
    def imag(self):
        """ Return the imaginary part of the complex literal """
        return self._imag_part

#------------------------------------------------------------------------------
class LiteralImaginaryUnit(LiteralComplex, Literal):
    """Represents the python value j"""
    def __new__(cls):
        return LiteralComplex.__new__(cls, 0, 1)

    def __init__(self):
        LiteralComplex.__init__(self, 0, 1)

#------------------------------------------------------------------------------
class LiteralString(Basic, Literal):
    """Represents a string literal in python"""
    _dtype     = NativeString()
    _precision = 0
    def __new__(cls, arg):
        return Basic.__new__(cls, arg)

    def __init__(self, arg):
        Basic.__init__(self)
        if not isinstance(arg, str):
            raise TypeError('arg must be of type str')
        self._string = arg

    @property
    def arg(self):
        """ Return the python string literal """
        return self._string

    def __str__(self):
        return self.arg

#------------------------------------------------------------------------------

def get_default_literal_value(dtype):
    """Returns the default value of a native datatype."""
    if isinstance(dtype, NativeInteger):
        value = LiteralInteger(0)
    elif isinstance(dtype, NativeReal):
        value = LiteralFloat(0.0)
    elif isinstance(dtype, NativeComplex):
        value = LiteralComplex(0.0, 0.0)
    elif isinstance(dtype, NativeBool):
        value = LiteralFalse()
    elif isinstance(dtype, NativeString):
        value = LiteralString('')
    else:
        raise TypeError('Unknown type')
    return value

