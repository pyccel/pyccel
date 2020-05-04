from .basic import PyccelAstNode
from sympy import Integer as sp_Integer
from sympy import Float as sp_Float
from sympy.logic.boolalg      import BooleanTrue as sp_BooleanTrue, BooleanFalse as sp_BooleanFalse
from sympy.core.expr          import Expr
from sympy.core.numbers       import NegativeOne as sp_NegativeOne
from .datatypes import default_precision

__all__ = ('BooleanTrue',
        'BooleanFalse',
        'Integer',
        'One',
        'NegativeOne',
        'Zero',
        'Float')

class BooleanTrue(sp_BooleanTrue, PyccelAstNode):
    _dtype     = 'bool'
    _rank      = 0
    _precision = default_precision['bool']

class BooleanFalse(sp_BooleanFalse, PyccelAstNode):
    _dtype     = 'bool'
    _rank      = 0
    _precision = default_precision['bool']

class Integer(sp_Integer, PyccelAstNode):
    _dtype     = 'int'
    _rank      = 0
    _precision = default_precision['int']
    def __new__(cls, val):
        val = int(val)
        if val == 0:
            return Zero()
        elif val == 1:
            return One()
        elif val == -1:
            return NegativeOne()
        else:
            return sp_Integer.__new__(cls, val)

class One(Integer):
    _dtype     = 'int'
    _rank      = 0
    _precision = default_precision['int']
    p          = 1
    def __new__(cls):
        return Expr.__new__(cls)

class NegativeOne(Integer, sp_NegativeOne):
    _dtype     = 'int'
    _rank      = 0
    _precision = default_precision['int']
    p          = -1
    def __new__(cls):
        return Expr.__new__(cls)

class Zero(Integer):
    _dtype     = 'int'
    _rank      = 0
    _precision = default_precision['int']
    p          = 0
    def __new__(cls):
        return Expr.__new__(cls)

class Float(sp_Float, PyccelAstNode):
    _dtype     = 'real'
    _rank      = 0
    _precision = default_precision['real']

