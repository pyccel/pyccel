#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
""" Module containing types from the numpy module understood by pyccel
"""
from functools import lru_cache

import numpy as np

from pyccel.utilities.metaclasses import ArgumentSingleton
from pyccel.utilities.stage   import PyccelStage

from .datatypes import FixedSizeNumericType, HomogeneousContainerType, PythonNativeBool
from .datatypes import PrimitiveBooleanType, PrimitiveIntegerType, PrimitiveFloatingPointType, PrimitiveComplexType
from .datatypes import GenericType
from .datatypes import pyccel_type_to_original_type, original_type_to_pyccel_type

__all__ = (
        'NumpyIntType',
        'NumpyInt8Type',
        'NumpyInt16Type',
        'NumpyInt32Type',
        'NumpyInt64Type',
        'NumpyFloat32Type',
        'NumpyFloat64Type',
        'NumpyFloat128Type',
        'NumpyComplex64Type',
        'NumpyComplex128Type',
        'NumpyComplex256Type',
        'NumpyNDArrayType',
        'NumpyNumericType',
        )

pyccel_stage = PyccelStage()

primitive_type_precedence = [PrimitiveBooleanType(), PrimitiveIntegerType(), PrimitiveFloatingPointType(), PrimitiveComplexType()]

#==============================================================================

class NumpyNumericType(FixedSizeNumericType):
    """
    Base class representing a scalar numeric datatype defined in the numpy module.

    Base class representing a scalar numeric datatype defined in the numpy module.
    """
    __slots__ = ()

    @lru_cache
    def __add__(self, other):
        try:
            return original_type_to_pyccel_type[
                    np.result_type(pyccel_type_to_original_type[self](),
                            pyccel_type_to_original_type[other]()).type]
        except KeyError:
            return NotImplemented

    @lru_cache
    def __radd__(self, other):
        return self.__add__(other)

    def __eq__(self, other):
        if other is self:
            return True
        elif isinstance(other, NumpyNumericType):
            return False
        elif isinstance(other, FixedSizeNumericType):
            return other.primitive_type == self.primitive_type and \
                    other.precision == self.precision
        else:
            return NotImplemented

    def __hash__(self):
        return hash(f"numpy.{self}")

#==============================================================================

class NumpyIntType(NumpyNumericType):
    """
    Super class representing NumPy's integer types.

    Super class representing NumPy's integer types.
    """
    __slots__ = ()
    _primitive_type = PrimitiveIntegerType()

    @lru_cache
    def __and__(self, other):
        if isinstance(other, PythonNativeBool):
            return self
        elif isinstance(other, FixedSizeNumericType):
            precision = max(self.precision, other.precision)
            return numpy_precision_map[(self._primitive_type, precision)]
        else:
            return NotImplemented

    @lru_cache
    def __rand__(self, other):
        if isinstance(other, PythonNativeBool):
            return self
        elif isinstance(other, FixedSizeNumericType):
            precision = max(self.precision, other.precision)
            return numpy_precision_map[(self._primitive_type, precision)]
        else:
            return NotImplemented


class NumpyInt8Type(NumpyIntType):
    """
    Class representing NumPy's int8 type.

    Class representing NumPy's int8 type.
    """
    __slots__ = ()
    _name = 'numpy.int8'
    _precision = 1


class NumpyInt16Type(NumpyIntType):
    """
    Class representing NumPy's int16 type.

    Class representing NumPy's int16 type.
    """
    __slots__ = ()
    _name = 'numpy.int16'
    _precision = 2


class NumpyInt32Type(NumpyIntType):
    """
    Class representing NumPy's int32 type.

    Class representing NumPy's int32 type.
    """
    __slots__ = ()
    _name = 'numpy.int32'
    _precision = 4


class NumpyInt64Type(NumpyIntType):
    """
    Class representing NumPy's int64 type.

    Class representing NumPy's int64 type.
    """
    __slots__ = ()
    _name = 'numpy.int64'
    _precision = 8

#==============================================================================

class NumpyFloat32Type(NumpyNumericType):
    """
    Class representing NumPy's float32 type.

    Class representing NumPy's float32 type.
    """
    __slots__ = ()
    _name = 'numpy.float32'
    _primitive_type = PrimitiveFloatingPointType()
    _precision = 4


class NumpyFloat64Type(NumpyNumericType):
    """
    Class representing NumPy's float64 type.

    Class representing NumPy's float64 type.
    """
    __slots__ = ()
    _name = 'numpy.float64'
    _primitive_type = PrimitiveFloatingPointType()
    _precision = 8


class NumpyFloat128Type(NumpyNumericType):
    """
    Class representing NumPy's float128 type.

    Class representing NumPy's float128 type.
    """
    __slots__ = ()
    _name = 'numpy.float128'
    _primitive_type = PrimitiveFloatingPointType()
    _precision = 16

#==============================================================================

class NumpyComplex64Type(NumpyNumericType):
    """
    Class representing NumPy's complex64 type.

    Class representing NumPy's complex64 type.
    """
    __slots__ = ()
    _name = 'numpy.complex64'
    _primitive_type = PrimitiveComplexType()
    _precision = 4

    @property
    def element_type(self):
        """
        The type of an element of the complex.

        The type of an element of the complex. In other words, the type
        of the floats which comprise the complex type.
        """
        return NumpyFloat32Type()


class NumpyComplex128Type(NumpyNumericType):
    """
    Class representing NumPy's complex128 type.

    Class representing NumPy's complex128 type.
    """
    __slots__ = ()
    _name = 'numpy.complex128'
    _primitive_type = PrimitiveComplexType()
    _precision = 8

    @property
    def element_type(self):
        """
        The type of an element of the complex.

        The type of an element of the complex. In other words, the type
        of the floats which comprise the complex type.
        """
        return NumpyFloat64Type()


class NumpyComplex256Type(NumpyNumericType):
    """
    Class representing NumPy's complex256 type.

    Class representing NumPy's complex256 type.
    """
    __slots__ = ()
    _name = 'numpy.complex256'
    _primitive_type = PrimitiveComplexType()
    _precision = 16

    @property
    def element_type(self):
        """
        The type of an element of the complex.

        The type of an element of the complex. In other words, the type
        of the floats which comprise the complex type.
        """
        return NumpyFloat128Type()

#==============================================================================

class NumpyNDArrayType(HomogeneousContainerType, metaclass = ArgumentSingleton):
    """
    Class representing the NumPy ND array type.

    Class representing the NumPy ND array type.

    Parameters
    ----------
    dtype : NumpyNumericType | PythonNativeBool | GenericType
        The internal datatype of the object (GenericType is allowed for external
        libraries, e.g. MPI).
    """
    __slots__ = ('_element_type',)
    _name = 'numpy.ndarray'

    def __init__(self, dtype):
        if pyccel_stage == 'semantic':
            assert isinstance(dtype, (NumpyNumericType, PythonNativeBool, GenericType))
        self._element_type = dtype
        super().__init__()

    @lru_cache
    def __add__(self, other):
        test_type = np.zeros(1, dtype = pyccel_type_to_original_type[self.element_type])
        if isinstance(other, FixedSizeNumericType):
            comparison_type = pyccel_type_to_original_type[other]()
        elif isinstance(other, NumpyNDArrayType):
            comparison_type = np.zeros(1, dtype = pyccel_type_to_original_type[other.element_type])
        else:
            return NotImplemented
        result_type = original_type_to_pyccel_type[np.result_type(test_type, comparison_type).type]
        return NumpyNDArrayType(result_type)

    @lru_cache
    def __radd__(self, other):
        return self.__add__(other)

    @lru_cache
    def __and__(self, other):
        elem_type = self.element_type
        if isinstance(other, FixedSizeNumericType):
            return NumpyNDArrayType(elem_type and other)
        elif isinstance(other, NumpyNDArrayType):
            return NumpyNDArrayType(elem_type+other.element_type)
        else:
            return NotImplemented

    @lru_cache
    def __rand__(self, other):
        return self.__and__(other)

    def switch_basic_type(self, new_type):
        """
        Change the basic type to the new type.

        Change the basic type to the new type. A new NumpyNDArrayType will be
        returned whose underlying elements are of the NumPy type which is
        equivalent to the new type (e.g. PythonNativeFloat may be replaced by
        np.float64).

        Parameters
        ----------
        new_type : PyccelType
            The new basic type.

        Returns
        -------
        PyccelType
            The new type.
        """
        assert isinstance(new_type, FixedSizeNumericType)
        new_type = numpy_precision_map[(new_type.primitive_type, new_type.precision)]
        cls = type(self)
        return cls(self.element_type.switch_basic_type(new_type))

#==============================================================================

numpy_precision_map = {
        (PrimitiveBooleanType(), -1): PythonNativeBool(),
        (PrimitiveIntegerType(), 1): NumpyInt8Type(),
        (PrimitiveIntegerType(), 2): NumpyInt16Type(),
        (PrimitiveIntegerType(), 4): NumpyInt32Type(),
        (PrimitiveIntegerType(), 8): NumpyInt64Type(),
        (PrimitiveFloatingPointType(), 4) : NumpyFloat32Type(),
        (PrimitiveFloatingPointType(), 8) : NumpyFloat64Type(),
        (PrimitiveFloatingPointType(), 16): NumpyFloat128Type(),
        (PrimitiveComplexType(), 4) : NumpyComplex64Type(),
        (PrimitiveComplexType(), 8) : NumpyComplex128Type(),
        (PrimitiveComplexType(), 16): NumpyComplex256Type(),
        }

numpy_type_to_original_type = {
    NumpyInt8Type()       : np.int8,
    NumpyInt16Type()      : np.int16,
    NumpyInt32Type()      : np.int32,
    NumpyInt64Type()      : np.int64,
    NumpyFloat32Type()    : np.float32,
    NumpyFloat64Type()    : np.float64,
    NumpyComplex64Type()  : np.complex64,
    NumpyComplex128Type() : np.complex128,
    }

# Large types don't exist on all systems
if hasattr(np, 'float128'):
    numpy_type_to_original_type.update({
        NumpyFloat128Type()   : np.float128,
        NumpyComplex256Type() : np.complex256,
        })

pyccel_type_to_original_type.update(numpy_type_to_original_type)
original_type_to_pyccel_type.update({v:k for k,v in numpy_type_to_original_type.items()})
original_type_to_pyccel_type[np.bool_] = PythonNativeBool()
