# coding: utf-8


from .core import PyccelPow, PyccelAdd, PyccelMul, PyccelDiv, PyccelMod, PyccelFloorDiv
from .core import PyccelEq,  PyccelNe,  PyccelLt,  PyccelLe,  PyccelGt,  PyccelGe
from .core import PyccelAnd, PyccelOr,  PyccelNot, Is, IsNot, PyccelAssociativeParenthesis
from .core import Variable, IndexedElement, DottedVariable

from .numbers   import Integer, Float, BooleanFalse, BooleanTrue
from .datatypes import NativeInteger, NativeReal, NativeComplex, NativeBool

from sympy.core.function import Application

__all__ = ('sp_dtype', 'str_dtype')

def sp_dtype(expr):
    """
    return the datatype of a sympy types expression

    """

    if isinstance(expr,(PyccelEq,PyccelNe, PyccelLt, PyccelLe,
                          PyccelGt, PyccelGe, PyccelAnd, PyccelOr,
                          PyccelNot, Is, IsNot)):
        return 'bool'
    elif isinstance(expr, (PyccelPow, PyccelAdd, PyccelMul, PyccelMod, 
                           PyccelFloorDiv, PyccelAssociativeParenthesis)):
        args       = [sp_dtype(a) for a in expr.args]
        is_integer = all(a=='integer' for a in args)
        is_real    = all(a=='integer' or a=='real' for a in args)
        is_complex = all(a=='integer' or a=='real' or a=='complex' for a in args)
        is_bool    = any(a=='bool' for a in args)
        if is_integer:
            return 'integer'
        elif is_real:
            return 'real'
        elif is_complex:
            return 'complex'
        elif is_bool:
            return 'bool'
    elif isinstance(expr, PyccelDiv):
        args       = [sp_dtype(a) for a in expr.args]

        is_real    = all(a=='integer' or a=='real' for a in args)
        is_complex = all(a=='integer' or a=='real' or a=='complex' for a in args)
        if is_real:
            return 'real'
        elif is_complex:
            return 'complex'

    elif isinstance(expr, (Variable, IndexedElement, DottedVariable)):
        return str_dtype(expr.dtype)
    elif isinstance(expr, Integer):
        return 'integer'
    elif isinstance(expr, Float):
        return 'real'
    elif isinstance(expr, (BooleanFalse, BooleanTrue)):
        return 'bool'
    elif isinstance(expr, Application):
        if expr.is_integer:
            return 'integer'
        elif expr.is_real:
            return 'real'
        elif expr.is_complex:
            return 'complex'
        elif expr.is_Boolean:
            return 'bool'
    raise TypeError('Unknown datatype {0}'.format(type(expr)))


def str_dtype(dtype):

    """
    return a sympy datatype as string
    dtype: str, Native Type

    """
    if isinstance(dtype, str):
        if dtype == 'int':
            return 'integer'
        elif dtype== 'real':
            return 'real'
        else:
            return dtype
    if isinstance(dtype, NativeInteger):
        return 'integer'
    elif isinstance(dtype, NativeReal):
        return 'real'
    elif isinstance(dtype, NativeComplex):
        return 'complex'
    elif isinstance(dtype, NativeBool):
        return 'bool'
    else:
        raise TypeError('Unknown datatype {0}'.format(str(dtype)))
