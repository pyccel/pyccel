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

from .datatypes      import VoidType
from .core           import Module, PyccelFunctionDef
from .internals      import PyccelFunction
from .internals      import LiteralInteger
from .numpyext       import process_dtype, process_shape , DtypePrecisionToCastFunction
from .numpytypes     import NumpyNDArrayType



__all__ = (
    'CudaSynchronize',
    'CudaNewarray'
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
    __slots__ = ('class_type', 'init_dtype', 'memory_location')

    def __init__(self, class_type, init_dtype, memory_location):
        self.class_type = class_type
        self.init_dtype = init_dtype
        self.memory_location = memory_location

        super().__init__()
    @staticmethod
    def _process_order(rank, order):

        if rank < 2:
            return None
        order = str(order).strip('\'"')
        assert order in ('C', 'F')
        return order

class CudaFull(CudaNewarray):
  
    __slots__ = ('_fill_value','_shape')

    def __init__(self, shape, fill_value, dtype='float', order='C'):
        shape = process_shape(False, shape)
        init_dtype = dtype
        if(dtype is None):
            dtype = fill_value.dtype

        dtype = process_dtype(dtype)

        # if fill_value and fill_value.dtype != dtype:
        #     cast_func = DtypePrecisionToCastFunction[dtype]
        #     fill_value = cast_func(fill_value)
        self.shape = shape
        rank = len(shape)
        order = CudaNewarray._process_order(rank, order)
        class_type = NumpyNDArrayType(dtype, shape, order)

        super().__init__(fill_value, class_type = class_type, init_dtype = init_dtype)


class CudaAutoFill(CudaFull):
    """ Abstract class for all classes which inherit from NumpyFull but
        the fill_value is implicitly specified
    """
    __slots__ = ()
    def __init__(self, shape, dtype='float', order='C'):
        if not dtype:
            raise TypeError("Data type must be provided")
        super().__init__(shape, None, dtype, order)

class CudaEmpty(CudaNewarray):
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

    def __init__(self, shape, dtype='float', order='C'):
        super().__init__(shape, dtype, order)


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
    'empty'             : PyccelFunctionDef('empty' , CudaEmpty),
}

cuda_mod = Module('cuda',
    variables=[],
    funcs=cuda_funcs.values(),
    imports=[]
)

