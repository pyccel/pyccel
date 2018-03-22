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

class Zeros(Function):
    """Represents a call to numpy.zeros for code generation.

    shape : int, list, tuple
        int or list of integers

    dtype: str, DataType
        datatype for the constructed array

    Examples

    """
    # TODO improve
    def __new__(cls, shape, dtype=None):

        if isinstance(shape, list):
            # this is a correction. otherwise it is not working on LRZ
            if isinstance(shape[0], list):
                shape = Tuple(*(sympify(i) for i in shape[0]))
            else:
                shape = Tuple(*(sympify(i) for i in shape))
        elif isinstance(shape, int):
            shape = Tuple(sympify(shape))
        else:
            shape = shape

        if dtype is None:
            dtype = 'double'

        if isinstance(dtype, str):
            dtype = datatype('ndarray'+dtype)
        elif not isinstance(dtype, DataType):
            raise TypeError("datatype must be an instance of DataType.")

        return Basic.__new__(cls, shape, dtype)

    @property
    def shape(self):
        return self._args[0]

    @property
    def rank(self):
        if iterable(self.shape):
            return len(self.shape)
        else:
            return 1

    @property
    def dtype(self):
        return self._args[1]

    @property
    def init_value(self):
        dtype = self.dtype
        if isinstance(dtype, NativeInteger):
            value = 0
        elif isinstance(dtype, NativeFloat):
            value = 0.0
        elif isinstance(dtype, NativeDouble):
            value = 0.0
        elif isinstance(dtype, NativeComplex):
            value = 0.0
        elif isinstance(dtype, NativeBool):
            value = BooleanFalse()
        else:
            raise TypeError('Unknown type')
        return value

    def fprint(self, printer, lhs):
        """Fortran print."""
        if isinstance(self.shape, Tuple):
            # this is a correction. problem on LRZ
            shape_code = ', '.join('0:' + printer(i-1) for i in self.shape)
        else:
            shape_code = '0:' + printer(self.shape-1)

        init_value = printer(self.init_value)

        lhs_code = printer(lhs)

        code_alloc = "allocate({0}({1}))".format(lhs_code, shape_code)
        code_init = "{0} = {1}".format(lhs_code, init_value)
        code = "{0}\n{1}".format(code_alloc, code_init)
        return code


class Ones(Zeros):
    """Represents a call to numpy.ones for code generation.

    shape : int or list of integers

    Examples

    """
    @property
    def init_value(self):
        dtype = self.dtype
        if isinstance(dtype, NativeInteger):
            value = 1
        elif isinstance(dtype, NativeFloat):
            value = 1.0
        elif isinstance(dtype, NativeDouble):
            value = 1.0
        elif isinstance(dtype, NativeComplex):
            value = 1.0
        elif isinstance(dtype, NativeBool):
            value = BooleanTrue()
        else:
            raise TypeError('Unknown type')
        return value
