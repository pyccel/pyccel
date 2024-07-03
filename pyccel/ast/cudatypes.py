#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
""" Module containing types from the numpy module understood by pyccel
"""
from functools import lru_cache
import numpy as np

from .datatypes import FixedSizeNumericType, HomogeneousContainerType, PythonNativeBool
from pyccel.utilities.metaclasses import ArgumentSingleton
from .datatypes import pyccel_type_to_original_type, original_type_to_pyccel_type


class CudaArrayType(HomogeneousContainerType, metaclass = ArgumentSingleton):
    """
    Class representing the Cuda array type.

    Class representing the Cuda array type

    dtype : NumpyNumericType | PythonNativeBool | GenericType
        The internal datatype of the object (GenericType is allowed for external
        libraries, e.g. MPI).
    rank : int
        The rank of the new NumPy array.
    order : str
        The order of the memory layout for the new NumPy array.
    memory_location : str
        The memory location of the new cuda array.
    """
    __slots__ = ('_dtype', '_rank', '_order', '_memory_location')

    # def __new__(cls, dtype, rank, order, memory_location):
    #     if rank == 0:
    #         return dtype
    #     else:
    #         return super().__new__(cls, dtype, rank, order)
    def __init__(self, dtype, rank, order, memory_location):
        assert isinstance(rank, int)
        assert order in (None, 'C', 'F')

        self._dtype = dtype
        self._rank = rank
        self._order = order
        self._memory_location = memory_location
        super().__init__()

    @lru_cache
    def __add__(self, other):
        test_type = np.zeros(1, dtype = pyccel_type_to_original_type[self.element_type])
        if isinstance(other, FixedSizeNumericType):
            comparison_type = pyccel_type_to_original_type[other]()
        elif isinstance(other, CudaArrayType):
            comparison_type = np.zeros(1, dtype = pyccel_type_to_original_type[other.element_type])
        else:
            return NotImplemented
        # Todo need to check for memory location as well
        result_type = original_type_to_pyccel_type[np.result_type(test_type, comparison_type).type]
        rank = max(other.rank, self.rank)
        if rank < 2:
            order = None
        else:
            other_f_contiguous = other.order in (None, 'F')
            self_f_contiguous = self.order in (None, 'F')
            order = 'F' if other_f_contiguous and self_f_contiguous else 'C'
        return CudaArrayType(result_type, rank, order, self.memory_location)

    @lru_cache
    def __radd__(self, other):
        return self.__add__(other)

    @lru_cache
    def __and__(self, other):
        elem_type = self.element_type
        if isinstance(other, FixedSizeNumericType):
            return CudaArrayType(elem_type and other)
        elif isinstance(other, CudaArrayType):
            return CudaArrayType(elem_type+other.element_type)
        else:
            return NotImplemented

    @lru_cache
    def __rand__(self, other):
        return self.__and__(other)

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
