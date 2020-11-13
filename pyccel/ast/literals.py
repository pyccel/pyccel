""" This module contains all literal types
"""
from .basic import PyccelAstNode
from sympy import Integer as sp_Integer
from sympy import Float as sp_Float
from sympy.logic.boolalg      import BooleanTrue as sp_BooleanTrue, BooleanFalse as sp_BooleanFalse
from sympy.core.expr          import Expr
from pyccel.ast.datatypes     import (NativeInteger, NativeBool, NativeReal,
                                  NativeComplex, default_precision)

__all__ = (
    'LiteralBooleanTrue',
    'LiteralBooleanFalse',
    'LiteralInteger',
    'LiteralFloat',
    'LiteralComplex',
    'get_default_value'
)

#------------------------------------------------------------------------------
class LiteralBooleanTrue(sp_BooleanTrue, PyccelAstNode):
    _dtype     = NativeBool()
    _rank      = 0
    _shape     = ()
    _precision = default_precision['bool']

#------------------------------------------------------------------------------
class LiteralBooleanFalse(sp_BooleanFalse, PyccelAstNode):
    _dtype     = NativeBool()
    _rank      = 0
    _shape     = ()
    _precision = default_precision['bool']

#------------------------------------------------------------------------------
class LiteralInteger(sp_Integer, PyccelAstNode):
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
    _dtype     = NativeReal()
    _rank      = 0
    _shape     = ()
    _precision = default_precision['real']

#------------------------------------------------------------------------------
class LiteralComplex(Expr, PyccelAstNode):
    _dtype     = NativeComplex()
    _rank      = 0
    _shape     = ()
    _precision = default_precision['complex']

    @property
    def real(self):
        return self.args[0]

    @property
    def imag(self):
        return self.args[1]

#------------------------------------------------------------------------------
class LiteralImaginaryUnit(Expr, PyccelAstNode):
    _dtype     = NativeComplex()
    _rank      = 0
    _shape     = ()
    _precision = default_precision['complex']

#------------------------------------------------------------------------------

def get_default_value(dtype):
    """Returns the default value of a native datatype."""
    if isinstance(dtype, NativeInteger):
        value = LiteralInteger(0)
    elif isinstance(dtype, NativeReal):
        value = LiteralFloat(0.0)
    elif isinstance(dtype, NativeComplex):
        value = LiteralComplex(0.0)
    elif isinstance(dtype, NativeBool):
        value = LiteralBooleanFalse()
    else:
        raise TypeError('Unknown type')
    return value

