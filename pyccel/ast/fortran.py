# coding: utf-8

# TODO remove sympify, Symbol

from sympy.core.function import Function
from sympy.core import Symbol, Tuple
from sympy import sympify
from sympy.core.basic import Basic
from sympy.utilities.iterables import iterable
from sympy.logic.boolalg import Boolean, BooleanTrue, BooleanFalse

from .core import Variable, IndexedElement, IndexedVariable
from .core import DataType, datatype
from .core import (NativeInteger, NativeFloat, NativeDouble, NativeComplex,
                   NativeBool)

# TODO: - implement all the following objects
class Ceil(Function):
    pass


