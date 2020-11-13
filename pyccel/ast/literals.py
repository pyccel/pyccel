""" This module contains all literal types
"""
from sympy               import Integer as sp_Integer
from sympy               import Float as sp_Float
from sympy.logic.boolalg import BooleanTrue as sp_BooleanTrue, BooleanFalse as sp_BooleanFalse
from sympy.core.expr     import Expr

from .basic              import PyccelAstNode, Basic
from .datatypes          import (NativeInteger, NativeBool, NativeReal,
                                  NativeComplex, NativeString, default_precision)

__all__ = (
    'LiteralBooleanTrue',
    'LiteralBooleanFalse',
    'LiteralInteger',
    'LiteralFloat',
    'LiteralComplex',
    'LiteralImaginaryUnit',
    'LiteralString',
    'get_default_literal_value'
)

#------------------------------------------------------------------------------
class LiteralBooleanTrue(sp_BooleanTrue, PyccelAstNode):
    """Represents the python value True"""
    _dtype     = NativeBool()
    _rank      = 0
    _shape     = ()
    _precision = default_precision['bool']

#------------------------------------------------------------------------------
class LiteralBooleanFalse(sp_BooleanFalse, PyccelAstNode):
    """Represents the python value False"""
    _dtype     = NativeBool()
    _rank      = 0
    _shape     = ()
    _precision = default_precision['bool']

#------------------------------------------------------------------------------
class LiteralInteger(sp_Integer, PyccelAstNode):
    """Represents an integer literal in python"""
    _dtype     = NativeInteger()
    _rank      = 0
    _shape     = ()
    _precision = default_precision['int']
    def __new__(cls, val):
        ival = int(val)
        obj = Expr.__new__(cls, ival)
        obj.p = ival
        return obj

#------------------------------------------------------------------------------
class LiteralFloat(sp_Float, PyccelAstNode):
    """Represents a float literal in python"""
    _dtype     = NativeReal()
    _rank      = 0
    _shape     = ()
    _precision = default_precision['real']

#------------------------------------------------------------------------------
class LiteralComplex(Expr, PyccelAstNode):
    """Represents a complex literal in python"""
    _dtype     = NativeComplex()
    _rank      = 0
    _shape     = ()
    _precision = default_precision['complex']

    def __new__(cls, real, imag):
        return Expr.__new__(cls)

    def __init__(self, real, imag):
        self._real_part = real
        self._imag_part = imag

    @property
    def real(self):
        """ Return the real part of the complex literal """
        return self._real_part

    @property
    def imag(self):
        """ Return the imaginary part of the complex literal """
        return self._imag_part

#------------------------------------------------------------------------------
class LiteralImaginaryUnit(Expr, PyccelAstNode):
    """Represents the python value j"""
    _dtype     = NativeComplex()
    _rank      = 0
    _shape     = ()
    _precision = default_precision['complex']

#------------------------------------------------------------------------------
class LiteralString(Basic, PyccelAstNode):
    """Represents a string literal in python"""
    _rank      = 0
    _shape     = ()
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
        value = LiteralBooleanFalse()
    elif isinstance(dtype, NativeString):
        value = LiteralString('')
    else:
        raise TypeError('Unknown type')
    return value

