""" This module contains all literal types
"""
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
    def __new__(cls, precision = default_precision['bool']):
        return sp_BooleanTrue.__new__(cls)
    def __init__(self, precision = default_precision['bool']):
        Literal.__init__(self, precision)

#------------------------------------------------------------------------------
class LiteralFalse(sp_BooleanFalse, Literal):
    """Represents the python value False"""
    _dtype     = NativeBool()
    def __new__(cls, precision = default_precision['bool']):
        return sp_BooleanFalse.__new__(cls)
    def __init__(self,precision = default_precision['bool']):
        Literal.__init__(self, precision)

#------------------------------------------------------------------------------
class LiteralInteger(Basic, Literal):
    """Represents an integer literal in python"""
    _dtype     = NativeInteger()
    def __new__(cls, value, precision = default_precision['integer']):
        return Basic.__new__(cls, value)

    def __init__(self, value, precision = default_precision['integer']):
        Literal.__init__(self, precision)
        if not isinstance(value, int):
            raise TypeError("A LiteralInteger can only be created with an integer")
        self.p = value

#------------------------------------------------------------------------------
class LiteralFloat(sp_Float, Literal):
    """Represents a float literal in python"""
    _dtype     = NativeReal()
    def __new__(cls, value, *, precision = default_precision['float']):
        return sp_Float.__new__(cls, value)

    def __init__(self, value, *, precision = default_precision['float']):
        if not isinstance(value, (int, float, LiteralFloat)):
            raise TypeError("A LiteralFloat can only be created with an integer or a float")
        Literal.__init__(self, precision)


#------------------------------------------------------------------------------
class LiteralComplex(Basic, Literal):
    """Represents a complex literal in python"""
    _dtype     = NativeComplex()

    def __new__(cls, real, imag, precision = default_precision['complex']):
        return Basic.__new__(cls, real, imag)

    def __init__(self, real, imag, precision = default_precision['complex']):
        Basic.__init__(self)
        Literal.__init__(self, precision)
        if isinstance(real, LiteralFloat):
            if real.precision == precision:
                self._real_part = real
            else:
                self._real_part = LiteralFloat(real.args[0], precision = precision)
        elif isinstance(real, LiteralInteger):
            self._real_part = LiteralFloat(real.p, precision = precision)
        elif isinstance(real, (int, float)):
            self._real_part = LiteralFloat(real, precision = precision)
        else:
            raise TypeError("The real part of a LiteralComplex must be an int/float/LiteralFloat")

        if isinstance(imag, LiteralFloat):
            if imag.precision == precision:
                self._imag_part = imag
            else:
                self._imag_part = LiteralFloat(imag.args[0], precision = precision)
        elif isinstance(imag, LiteralInteger):
            self._imag_part = LiteralFloat(imag.p, precision = precision)
        elif isinstance(imag, (int, float)):
            self._imag_part = LiteralFloat(imag, precision = precision)
        else:
            raise TypeError("The imaginary part of a LiteralComplex must be an int/float/LiteralFloat")

    @property
    def real(self):
        """ Return the real part of the complex literal """
        return self._real_part

    @property
    def imag(self):
        """ Return the imaginary part of the complex literal """
        return self._imag_part

#------------------------------------------------------------------------------
class LiteralImaginaryUnit(LiteralComplex):
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

