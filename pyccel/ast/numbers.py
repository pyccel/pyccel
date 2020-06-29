from .basic import PyccelAstNode
from sympy import Integer as sp_Integer
from sympy import Float as sp_Float
from sympy.logic.boolalg      import BooleanTrue as sp_BooleanTrue, BooleanFalse as sp_BooleanFalse
from sympy.core.expr          import Expr
from sympy.core.numbers       import NegativeOne as sp_NegativeOne
from pyccel.ast.datatypes     import (datatype, DataType, NativeSymbol,
                                  NativeInteger, NativeBool, NativeReal,
                                  NativeComplex, NativeRange, NativeTensor, NativeString,
                                  NativeGeneric, NativeTuple, default_precision)

__all__ = (
    'BooleanTrue',
    'BooleanFalse',
    'Integer',
    'Float',
    'Complex',
)

#------------------------------------------------------------------------------
class BooleanTrue(sp_BooleanTrue, PyccelAstNode):
    _dtype     = NativeBool()
    _rank      = 0
    _shape     = ()
    _precision = default_precision['bool']

#------------------------------------------------------------------------------
class BooleanFalse(sp_BooleanFalse, PyccelAstNode):
    _dtype     = NativeBool()
    _rank      = 0
    _shape     = ()
    _precision = default_precision['bool']

#------------------------------------------------------------------------------
class Integer(sp_Integer, PyccelAstNode):
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
class Float(sp_Float, PyccelAstNode):
    _dtype     = NativeReal()
    _rank      = 0
    _shape     = ()
    _precision = default_precision['real']

#------------------------------------------------------------------------------
class Complex(Expr, PyccelAstNode):
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
