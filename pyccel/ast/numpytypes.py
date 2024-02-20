#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
""" Module containing types from the numpy module understood by pyccel
"""

from .datatypes import FixedSizeNumericType, HomogeneousContainerType, ComplexType
from .datatypes import PyccelBooleanType, PyccelIntegerType, PyccelFloatingPointType
from .type_annotations import typenames_to_dtypes

__all__ = (
        'NumpyInt8Type',
        'NumpyInt16Type',
        'NumpyInt32Type',
        'NumpyInt64Type',
        'NumpyFloat32Type',
        'NumpyFloat64Type',
        'NumpyNDArrayType',
        'NumpyNumericType',
        )

primitive_type_precedence = [PyccelBooleanType(), PyccelIntegerType(), PyccelFloatingPointType()]

numpy_precision_map = {
        (PyccelIntegerType(), 1): NumpyInt8Type(),
        (PyccelIntegerType(), 2): NumpyInt16Type(),
        (PyccelIntegerType(), 4): NumpyInt32Type(),
        (PyccelIntegerType(), 8): NumpyInt64Type(),
        (PyccelFloatingPointType(), 4): NumpyFloat32Type(),
        (PyccelFloatingPointType(), 8): NumpyFloat64Type(),
        }

typenames_to_dtypes.update({
    'int8' : NumpyInt8Type(),
    'int16' : NumpyInt16Type(),
    'int32' : NumpyInt32Type(),
    'int64' : NumpyInt64Type(),
    'i1' : NumpyInt8Type(),
    'i2' : NumpyInt16Type(),
    'i4' : NumpyInt32Type(),
    'i8' : NumpyInt64Type(),
    'float32' : NumpyFloat32Type(),
    'float64' : NumpyFloat32Type(),
    'f4' : NumpyFloat32Type(),
    'f8' : NumpyFloat32Type(),
    'complex64' : ComplexType(NumpyFloat32Type()),
    'complex128' : ComplexType(NumpyFloat64Type()),
    'c8' : ComplexType(NumpyFloat32Type()),
    'c16' : ComplexType(NumpyFloat64Type()),
    })

#==============================================================================

class NumpyNumericType(FixedSizeNumericType):
    """
    Base class representing a scalar numeric datatype defined in the numpy module.

    Base class representing a scalar numeric datatype defined in the numpy module.
    """
    __slots__ = ()

    @lru_cache
    def __add__(self, other):
        if isinstance(other, PythonNativeBool):
            return self
        elif isinstance(other, ComplexType):
            return ComplexType(self + other.element_type)
        elif isinstance(other, FixedSizeNumericType):
            primitive_dtype = primitive_type_precedence[max(primitive_type_precedence.index(self.primitive_type),
                                                            primitive_type_precedence.index(other.primitive_type))]
            precision = max(self.precision, other.precision)
            return numpy_precision_map[(primitive_dtype, precision)]
        else:
            return NotImplemented

    @lru_cache
    def __radd__(self, other):
        if isinstance(other, PythonNativeBool):
            return self
        elif isinstance(other, ComplexType):
            return ComplexType(self + other.element_type)
        elif isinstance(other, FixedSizeNumericType):
            primitive_dtype = primitive_type_precedence[max(primitive_type_precedence.index(self.primitive_type),
                                                            primitive_type_precedence.index(other.primitive_type))]
            precision = max(self.precision, other.precision)
            return numpy_precision_map[(primitive_dtype, precision)]
        else:
            return NotImplemented

class NumpyInt8Type(NumpyNumericType):
    """
    Class representing NumPy's int8 type.

    Class representing NumPy's int8 type.
    """
    __slots__ = ()
    _name = 'int8'
    _primitive_type = PyccelIntegerType()
    _precision = 1

class NumpyInt16Type(NumpyNumericType):
    """
    Class representing NumPy's int16 type.

    Class representing NumPy's int16 type.
    """
    __slots__ = ()
    _name = 'int16'
    _primitive_type = PyccelIntegerType()
    _precision = 2

class NumpyInt32Type(NumpyNumericType):
    """
    Class representing NumPy's int32 type.

    Class representing NumPy's int32 type.
    """
    __slots__ = ()
    _name = 'int32'
    _primitive_type = PyccelIntegerType()
    _precision = 4

class NumpyInt64Type(NumpyNumericType):
    """
    Class representing NumPy's int64 type.

    Class representing NumPy's int64 type.
    """
    __slots__ = ()
    _name = 'int64'
    _primitive_type = PyccelIntegerType()
    _precision = 8

#==============================================================================

class NumpyFloat32Type(NumpyNumericType):
    """
    Class representing NumPy's float32 type.

    Class representing NumPy's float32 type.
    """
    __slots__ = ()
    _name = 'float32'
    _primitive_type = PyccelFloatingPointType()
    _precision = 4

class NumpyFloat64Type(NumpyNumericType):
    """
    Class representing NumPy's float64 type.

    Class representing NumPy's float64 type.
    """
    __slots__ = ()
    _name = 'float64'
    _primitive_type = PyccelFloatingPointType()
    _precision = 8

#==============================================================================

class NumpyNDArrayType(HomogeneousContainerType, metaclass=Singleton):
    """
    Class representing the NumPy ND array type.

    Class representing the NumPy ND array type.
    """
    __slots__ = ()
    name = 'numpy.ndarray'

    def __init__(self, dtype):
        assert isinstance(dtype, NumpyNumericType)
        super().__init__(dtype)

    @lru_cache
    def __add__(self, other):
        if isinstance(other, self.element_type):
            return self
        elif isinstance(other, ComplexType):
            return NumpyNDArrayType(other)
        elif isinstance(other, NumpyNumericType):
            elem_type = self.element_type
            if isinstance(elem_type, ComplexType):
                return self
            elif primitive_type_precedence.index(elem_type.primitive_type) > primitive_type_precedence.index(other.primitive_type):
                return self
            else:
                return NumpyNDArrayType(other)
        elif isinstance(other, NumpyNDArrayType):
            return NumpyNDArrayType(self.element_type+other.element_type
        else:
            return NotImplemented

    @lru_cache
    def __radd__(self, other):
        return self.__add__(other)
