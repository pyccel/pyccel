from functools import reduce
import operator

import numpy

from pyccel.errors.errors import Errors
from pyccel.errors.messages import WRONG_LINSPACE_ENDPOINT, NON_LITERAL_KEEP_DIMS, NON_LITERAL_AXIS

from pyccel.utilities.stage import PyccelStage

from .basic          import PyccelAstNode
from .builtins       import (PythonInt, PythonBool, PythonFloat, PythonTuple,
                             PythonComplex, PythonReal, PythonImag, PythonList,
                             PythonType, PythonConjugate)

from .core           import Module, Import, PyccelFunctionDef

from .datatypes      import (dtype_and_precision_registry as dtype_registry,
                             default_precision, datatype, NativeInteger,
                             NativeFloat, NativeComplex, NativeBool, str_dtype,
                             NativeNumeric)

from .internals      import PyccelInternalFunction, Slice, max_precision, get_final_precision
from .internals      import PyccelArraySize

from .literals       import LiteralInteger, LiteralFloat, LiteralComplex, convert_to_literal
from .literals       import LiteralTrue, LiteralFalse
from .literals       import Nil
from .mathext        import MathCeil
from .operators      import broadcast, PyccelMinus, PyccelDiv
from .variable       import (Variable, Constant, HomogeneousTupleVariable)

from .numpyext       import process_dtype, process_shape

#==============================================================================
__all__ = (
    'CudaMemCopy',
    'CudaNewArray',
    'CudaArray',
    'CudaDeviceSynchronize',
    'CudaInternalVar',
    'CudaThreadIdx',
    'CudaBlockDim',
    'CudaBlockIdx',
    'CudaGridDim'
)

#==============================================================================
class CudaNewArray(PyccelInternalFunction):
    """ Class from which all Cuda functions which imply a call to Allocate
    inherit
    """
    __slots__ = ()
    #--------------------------------------------------------------------------
    @staticmethod
    def _process_order(rank, order):

        if rank < 2:
            return None

        order = str(order).strip('\'"')
        if order not in ('C', 'F'):
            raise ValueError('unrecognized order = {}'.format(order))
        return order

#==============================================================================

#==============================================================================
class CudaArray(CudaNewArray):
    """
    Represents a call to  cuda.array for code generation.

    arg : list, tuple, PythonList

    """
    __slots__ = ('_arg','_dtype','_precision','_shape','_rank','_order','_memory_location')
    _attribute_nodes = ('_arg',)
    name = 'array'

    def __init__(self, arg, dtype=None, order='C', memory_location='managed'):

        if not isinstance(arg, (PythonTuple, PythonList, Variable)):
            raise TypeError('Unknown type of  %s.' % type(arg))

        is_homogeneous_tuple = isinstance(arg, (PythonTuple, PythonList, HomogeneousTupleVariable)) and arg.is_homogeneous
        is_array = isinstance(arg, Variable) and arg.is_ndarray

        # TODO: treat inhomogenous lists and tuples when they have mixed ordering
        if not (is_homogeneous_tuple or is_array):
            raise TypeError('we only accept homogeneous arguments')

        # Verify dtype and get precision
        if dtype is None:
            dtype = arg.dtype
            prec = get_final_precision(arg)
        else:
            dtype, prec = process_dtype(dtype)
        # ... Determine ordering
        order = str(order).strip("\'")

        shape = process_shape(False, arg.shape)
        rank  = len(shape)

        if rank < 2:
            order = None
        else:
            # ... Determine ordering
            order = str(order).strip("\'")

            if order not in ('K', 'A', 'C', 'F'):
                raise ValueError(f"Cannot recognize '{order}' order")

            # TODO [YG, 18.02.2020]: set correct order based on input array
            if order in ('K', 'A'):
                order = 'C'
            # ...
        #Verify memory location
        if memory_location not in ('host', 'device', 'managed'):
            raise ValueError("memory_location must be 'host', 'device' or 'managed'")
        self._arg   = arg
        self._shape = shape
        self._rank  = rank
        self._dtype = dtype
        self._order = order
        self._precision = prec
        self._memory_location = memory_location
        super().__init__()

    def __str__(self):
        return str(self.arg)

    @property
    def arg(self):
        return self._arg
    @property
    def memory_location(self):
        return self._memory_location

class CudaDeviceSynchronize(PyccelInternalFunction):
    "Represents a call to  Cuda.deviceSynchronize for code generation."
    # pass
    _attribute_nodes = ()
    def __init__(self):
        #...
        self._shape     = None
        self._rank      = 0
        self._dtype     = NativeInteger()
        self._precision = None
        self._order     = None
        super().__init__()

class CudaInternalVar(PyccelAstNode):
    _attribute_nodes = ('_dim',)

    def __init__(self, dim=None):
        if not isinstance(dim, LiteralInteger):
            raise TypeError("dimension need to be an integer")
        if dim not in (0, 1, 2):
            raise ValueError("dimension need to be 0, 1 or 2")
        #...
        self._dim       = dim
        self._shape     = None
        self._rank      = 0
        self._dtype     = dim.dtype
        self._precision = dim.precision
        self._order     = None
        super().__init__()

    @property
    def dim(self):
        return self._dim

class CudaThreadIdx(CudaInternalVar) : pass
class CudaBlockDim(CudaInternalVar)  : pass
class CudaBlockIdx(CudaInternalVar)  : pass
class CudaGridDim(CudaInternalVar)   : pass


cuda_funcs = {
    # 'deviceSynchronize' : CudaDeviceSynchronize,
    'array'             : PyccelFunctionDef('array'             , CudaArray),
    'deviceSynchronize' : PyccelFunctionDef('deviceSynchronize' , CudaDeviceSynchronize),
    'threadIdx'         : PyccelFunctionDef('threadIdx'         , CudaThreadIdx),
    'blockDim'          : PyccelFunctionDef('blockDim'          , CudaBlockDim),
    'blockIdx'          : PyccelFunctionDef('blockIdx'          , CudaBlockIdx),
    'gridDim'           : PyccelFunctionDef('gridDim'           , CudaGridDim)
}

cuda_Internal_Var = {
    'CudaThreadIdx' : 'threadIdx',
    'CudaBlockDim'  : 'blockDim',
    'CudaBlockIdx'  : 'blockIdx',
    'CudaGridDim'   : 'gridDim'
}

cuda_constants = {

}
cuda_mod = Module('cuda',
    variables = cuda_constants.values(),
    funcs = cuda_funcs.values())
