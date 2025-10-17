#!/usr/bin/python
# -*- coding: utf-8 -*-
#pylint: disable=no-member
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
""" Module containing types from the numpy module understood by pyccel
"""
from functools import lru_cache
from packaging.version import Version

import numpy as np

from pyccel.utilities.metaclasses import Singleton
from pyccel.utilities.stage   import PyccelStage

from .datatypes import FixedSizeNumericType, HomogeneousContainerType, PythonNativeBool
from .datatypes import PrimitiveBooleanType, PrimitiveIntegerType, PrimitiveFloatingPointType, PrimitiveComplexType
from .datatypes import GenericType, CharType
from .datatypes import pyccel_type_to_original_type, original_type_to_pyccel_type

__all__ = (
        'NumpyComplex64Type',
        'NumpyComplex128Type',
        'NumpyComplex256Type',
        'NumpyFloat32Type',
        'NumpyFloat64Type',
        'NumpyFloat128Type',
        'NumpyInt8Type',
        'NumpyInt16Type',
        'NumpyInt32Type',
        'NumpyInt64Type',
        'NumpyIntType',
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

class NumpyNDArrayType(HomogeneousContainerType, metaclass = Singleton):
    """
    Class representing the NumPy ND array type.

    Class representing the NumPy ND array type.
    """
    __slots__ = ('_element_type', '_container_rank', '_order')
    _name = 'numpy.ndarray'

    @classmethod
    @lru_cache
    def get_new(cls, dtype, rank, order):
        """
        Get the parametrised NumPy ND array type.

        Get the parametrised NumPy ND array type.

        Parameters
        ----------
        dtype : NumpyNumericType | PythonNativeBool | GenericType
            The internal datatype of the object (GenericType is allowed for external
            libraries, e.g. MPI).
        rank : int
            The rank of the new NumPy array.
        order : str
            The order of the memory layout for the new NumPy array.
        """
        assert isinstance(rank, int)
        assert order in (None, 'C', 'F')
        assert rank < 2 or order is not None
        assert isinstance(dtype, (NumpyNumericType, PythonNativeBool, GenericType, CharType))

        if rank == 0:
            return dtype

        def __init__(self):
            self._element_type = dtype
            self._container_rank = rank
            self._order = order
            super().__init__()

        name = 'Numpy{rank}DArrayType_{order}_{type(dtype)}'
        return type(name, (NumpyNDArrayType,),
                    {'__init__': __init__})()

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
        rank = max(other.rank, self.rank)
        if rank < 2:
            order = None
        else:
            other_f_contiguous = other.order in (None, 'F')
            self_f_contiguous = self.order in (None, 'F')
            order = 'F' if other_f_contiguous and self_f_contiguous else 'C'
        return NumpyNDArrayType.get_new(result_type, rank, order)

    @lru_cache
    def __radd__(self, other):
        return self.__add__(other)

    @lru_cache
    def __and__(self, other):
        elem_type = self.element_type
        if isinstance(other, FixedSizeNumericType):
            return self.switch_basic_type(elem_type & other)
        elif isinstance(other, NumpyNDArrayType):
            return self.switch_basic_type(elem_type & other.element_type)
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
        return cls.get_new(self.element_type.switch_basic_type(new_type), self._container_rank, self._order)

    def switch_rank(self, new_rank, new_order = None):
        """
        Get a type which is identical to this type in all aspects except the rank and/or order.

        Get a type which is identical to this type in all aspects except the rank and/or order.
        The order must be provided if the rank is increased from 1. Otherwise it defaults to the
        same order as the current type.

        Parameters
        ----------
        new_rank : int
            The rank of the new type.

        new_order : str, optional
            The order of the new type. This should be provided if the rank is increased from 1.

        Returns
        -------
        PyccelType
            The new type.
        """
        if new_rank == 0:
            return self.element_type
        else:
            new_order = (new_order or self._order) if new_rank > 1 else None
            return NumpyNDArrayType.get_new(self.element_type, new_rank, new_order)

    def swap_order(self):
        """
        Get a type which is identical to this type in all aspects except the order.

        Get a type which is identical to this type in all aspects except the order.
        In the case of a 1D array the final type will be the same as this type. Otherwise
        if the array is C-ordered the final type will be F-ordered, while if the array
        is F-ordered the final type will be C-ordered.

        Returns
        -------
        PyccelType
            The new type.
        """
        order = None if self._order is None else ('C' if self._order == 'F' else 'F')
        return NumpyNDArrayType.get_new(self.element_type, self._container_rank, order)

    @property
    def rank(self):
        """
        Number of dimensions of the object.

        Number of dimensions of the object. If the object is a scalar then
        this is equal to 0.
        """
        return self._container_rank

    @property
    def order(self):
        """
        The data layout ordering in memory.

        Indicates whether the data is stored in row-major ('C') or column-major
        ('F') format. This is only relevant if rank > 1. When it is not relevant
        this function returns None.
        """
        return self._order

    def __repr__(self):
        dims = ','.join(':'*self._container_rank)
        order_str = f'(order={self._order})' if self._order else ''
        return f'{self.element_type}[{dims}]{order_str}'

    def __hash__(self):
        return hash((self.element_type, self.rank, self.order))

    def __eq__(self, other):
        return isinstance(other, NumpyNDArrayType) and self.element_type == other.element_type \
                and self.rank == other.rank and self.order == other.order

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

if Version(np.__version__) >= Version("2.0.0"):
    NumpyInt = NumpyInt64Type()
else:
    NumpyInt = numpy_precision_map[PrimitiveIntegerType(), np.dtype(int).alignment]
