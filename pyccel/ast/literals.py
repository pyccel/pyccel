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

    @property
    def python_value(self):
        """ Get python literal represented by this instance """

    def __repr__(self):
        return repr(self.python_value)

    def _sympystr(self, printer):
        return printer.doprint(self.python_value)

#------------------------------------------------------------------------------
class LiteralTrue(Literal, sp_BooleanTrue):
    """Represents the python value True"""
    _dtype     = NativeBool()
    def __new__(cls, precision = default_precision['bool']):
        return sp_BooleanTrue.__new__(cls)
    def __init__(self, precision = default_precision['bool']):
        Literal.__init__(self, precision)

    @property
    def python_value(self):
        return True

#------------------------------------------------------------------------------
class LiteralFalse(Literal, sp_BooleanFalse):
    """Represents the python value False"""
    _dtype     = NativeBool()
    def __new__(cls, precision = default_precision['bool']):
        return sp_BooleanFalse.__new__(cls)
    def __init__(self,precision = default_precision['bool']):
        Literal.__init__(self, precision)

    @property
    def python_value(self):
        return False

#------------------------------------------------------------------------------
class LiteralInteger(Literal, Basic):
    """Represents an integer literal in python"""
    _dtype     = NativeInteger()
    def __new__(cls, value, precision = default_precision['integer']):
        return Basic.__new__(cls, value)

    def __init__(self, value, precision = default_precision['integer']):
        Basic.__init__(self)
        Literal.__init__(self, precision)
        if not isinstance(value, int):
            raise TypeError("A LiteralInteger can only be created with an integer")
        self.p = value

    @property
    def python_value(self):
        return self.p

#------------------------------------------------------------------------------
class LiteralFloat(Literal, sp_Float):
    """Represents a float literal in python"""
    _dtype     = NativeReal()
    def __new__(cls, value, *, precision = default_precision['float']):
        return sp_Float.__new__(cls, value)

    def __init__(self, value, *, precision = default_precision['float']):
        if not isinstance(value, (int, float, LiteralFloat)):
            raise TypeError("A LiteralFloat can only be created with an integer or a float")
        Literal.__init__(self, precision)

    @property
    def python_value(self):
        return float(self)


#------------------------------------------------------------------------------
class LiteralComplex(Literal, Basic):
    """Represents a complex literal in python"""
    _dtype     = NativeComplex()

    def __new__(cls, real, imag, precision = default_precision['complex']):
        if cls is LiteralImaginaryUnit:
            return Basic.__new__(cls, real, imag)
        real_part = cls._collect_python_val(real)
        imag_part = cls._collect_python_val(imag)
        if real_part == 0 and imag_part == 1:
            return LiteralImaginaryUnit()
        else:
            return Basic.__new__(cls, real, imag)

    def __init__(self, real, imag, precision = default_precision['complex']):
        Basic.__init__(self)
        Literal.__init__(self, precision)
        self._real_part = LiteralFloat(self._collect_python_val(real))
        self._imag_part = LiteralFloat(self._collect_python_val(imag))

    @staticmethod
    def _collect_python_val(arg):
        if isinstance(arg, Literal):
            return arg.python_value
        elif isinstance(arg, (int, float)):
            return arg
        else:
            raise TypeError("LiteralComplex argument must be an int/float/LiteralInt/LiteralFloat")

    @property
    def real(self):
        """ Return the real part of the complex literal """
        return self._real_part

    @property
    def imag(self):
        """ Return the imaginary part of the complex literal """
        return self._imag_part

    @property
    def python_value(self):
        return self.real.python_value + self.imag.python_value*1j

#------------------------------------------------------------------------------
class LiteralImaginaryUnit(LiteralComplex):
    """Represents the python value j"""
    def __new__(cls):
        return LiteralComplex.__new__(cls, 0, 1)

    def __init__(self, real=0, imag=1, precision = default_precision['complex']):
        LiteralComplex.__init__(self, 0, 1)

    @property
    def python_value(self):
        return 1j

#------------------------------------------------------------------------------
class LiteralString(Literal, Basic):
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

    def __repr__(self):
        return "'{}'".format(str(self.python_value))

    def __str__(self):
        return str(self.python_value)

    @property
    def python_value(self):
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

