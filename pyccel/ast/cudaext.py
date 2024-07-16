#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
CUDA Extension Module
Provides CUDA functionality for code generation.
"""
from .internals      import PyccelFunction
from .literals       import Nil

from .datatypes      import VoidType
from .core           import Module, PyccelFunctionDef
from .internals      import PyccelFunction
from .internals      import LiteralInteger
from .numpyext       import process_dtype, process_shape , DtypePrecisionToCastFunction
from .cudatypes      import CudaArrayType



__all__ = (
    'CudaSynchronize',
    'CudaNewarray'
    'CudaFull'
    'CudaEmpty'
)

class CudaNewarray(PyccelFunction):
    """
    superclass for nodes representing Cuda array allocation functions.

    Class from which all nodes representing a Cuda function which implies a call
    to `Allocate` should inherit.

    Parameters

    class_type : NumpyNDArrayType
        The type of the new array.

    init_dtype : PythonType, PyccelFunctionDef, LiteralString, str
        The actual dtype passed to the Cuda function.

    memory_location : str
        The memory location of the new array ('host' or 'device').
    """
    __slots__ = ('_class_type', '_init_dtype', '_memory_location')

    property
    def init_dtype(self):
        """
        The dtype provided to the function when it was initialised in Python.

        The dtype provided to the function when it was initialised in Python.
        If no dtype was provided then this should equal `None`.
        """
        return self._init_dtype

    def __init__(self, *arg,class_type, init_dtype, memory_location):
        self._class_type = class_type
        self._init_dtype = init_dtype
        self._memory_location = memory_location

        super().__init__(*arg)
    @staticmethod
    def _process_order(rank, order):

        if rank < 2:
            return None
        order = str(order).strip('\'"')
        assert order in ('C', 'F')
        return order

class CudaFull(CudaNewarray):
  
    __slots__ = ('_fill_value','_shape')
    name = 'full'

    def __init__(self, shape, fill_value, dtype, order, memory_location):
        shape = process_shape(False, shape)
        init_dtype = dtype
        if(dtype is None):
            dtype = fill_value.dtype

        dtype = process_dtype(dtype)

        self._shape = shape
        rank = len(self._shape)
        order = CudaNewarray._process_order(rank, order)
        class_type = CudaArrayType(dtype, rank, order, memory_location)
        super().__init__(fill_value, class_type = class_type, init_dtype = init_dtype, memory_location = memory_location)
    @property
    def fill_value(self):
        return self._args[0]

class CudaAutoFill(CudaFull):
    """ Abstract class for all classes which inherit from NumpyFull but
        the fill_value is implicitly specified
    """
    __slots__ = ()
    def __init__(self, shape, dtype, order, memory_location):
        super().__init__(shape, Nil(), dtype, order, memory_location = memory_location)

class CudaHostEmpty(CudaAutoFill):
    """
    Represents a call to  Cuda.host_empty for code generation.

    A class representing a call to the Cuda `host_empty` function.

    Parameters
    ----------
    shape : tuple of int , int
        The shape of the new array.

    dtype : PythonType, LiteralString, str
        The actual dtype passed to the NumPy function.

    order : str , LiteralString
        The order passed to the function defoulting to 'C'.
    """
    __slots__ = ()
    name = 'empty'
    def __init__(self, shape, dtype='float', order='C'):
        memory_location = 'host'
        super().__init__(shape, dtype, order , memory_location)
    
    @property
    def fill_value(self):
        """
        The value with which the array will be filled on initialisation.

        The value with which the array will be filled on initialisation.
        """
        return None

class CudaSynchronize(PyccelFunction):
    """
    Represents a call to Cuda.synchronize for code generation.

    This class serves as a representation of the Cuda.synchronize method.
    """
    __slots__ = ()
    _attribute_nodes = ()
    _shape     = None
    _class_type = VoidType()
    def __init__(self):
        super().__init__()

cuda_funcs = {
    'synchronize'       : PyccelFunctionDef('synchronize' , CudaSynchronize),
    'full'              : PyccelFunctionDef('full' , CudaFull),
    'host_empty'             : PyccelFunctionDef('host_empty' , CudaHostEmpty),
}

cuda_mod = Module('cuda',
    variables=[],
    funcs=cuda_funcs.values(),
    imports=[]
)

