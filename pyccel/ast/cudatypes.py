#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
""" Module containing types from the numpy module understood by pyccel
"""

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
    
    def __new__(cls, dtype, rank, order):
        if rank == 0:
            return dtype
        else:
            return super().__new__(cls, dtype, rank, order)
    
    
