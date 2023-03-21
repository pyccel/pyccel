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
                             NativeFloat, NativeComplex, NativeBool, NativeVoid, str_dtype,
                             NativeNumeric)

from .internals      import PyccelInternalFunction, Slice, max_precision, get_final_precision
from .internals      import PyccelArraySize

from .literals       import LiteralInteger, LiteralFloat, LiteralComplex, convert_to_literal
from .literals       import LiteralTrue, LiteralFalse
from .literals       import Nil
from .mathext        import MathCeil
from .operators      import PyccelAdd, PyccelLe, PyccelMul, broadcast, PyccelMinus, PyccelDiv
from .variable       import (Variable, Constant, HomogeneousTupleVariable)

from .numpyext       import process_dtype, process_shape

#==============================================================================
__all__ = (
    'CudaArray',
    'CudaBlockDim',
    'CudaBlockIdx',
    'CudaCopy',
    'CudaGrid',
    'CudaGridDim',
    'CudaInternalVar',
    'CudaMemCopy',
    'CudaNewArray',
    'CudaSynchronize',
    'CudaThreadIdx'
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

class CudaSynchronize(PyccelInternalFunction):
    "Represents a call to  Cuda.deviceSynchronize for code generation."

    __slots__ = ('_dtype','_precision','_shape','_rank','_order')
    _attribute_nodes = ()
    def __init__(self):
        #...
        self._shape     = None
        self._rank      = 0
        self._dtype     = NativeVoid()
        self._precision = 0
        self._order     = None
        super().__init__()

class CudaInternalVar(PyccelAstNode):
    """
    Represents a General Class For Cuda internal Variables Used To locate Thread In the GPU architecture"

    Parameters
    ----------
    dim : NativeInteger
        Represent the dimension where we want to locate our thread.

    """
    __slots__ = ('_dim','_dtype','_precision','_shape','_rank','_order')
    _attribute_nodes = ('_dim',)

    def __init__(self, dim=None):
        
        if isinstance(dim, int):
            dim = LiteralInteger(dim)
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


class CudaCopy(CudaNewArray):
    """
    Represents a call to  cuda.copy for code generation.

    Parameters
    ----------
    arg : Variable

    memory_location : str
        'host'   the newly created array is allocated on host.
        'device' the newly created array is allocated on device.
    
    is_async: bool
        Indicates whether the copy is asynchronous or not [Default value: False]

    """
    __slots__ = ('_arg','_dtype','_precision','_shape','_rank','_order','_memory_location', '_is_async')

    def __init__(self, arg, memory_location, is_async=False):
        
        if not isinstance(arg, Variable):
            raise TypeError('unknown type of  %s.' % type(arg))
        
        # Verify the memory_location of src
        if arg.memory_location not in ('device', 'host', 'managed'):
            raise ValueError("The direction of the copy should be from 'host' or 'device'")

        # Verify the memory_location of dst
        if memory_location not in ('device', 'host', 'managed'):
            raise ValueError("The direction of the copy should be to 'host' or 'device'")
        
        # verify the type of is_async
        if not isinstance(is_async, (LiteralTrue, LiteralFalse, bool)):
            raise TypeError('is_async must be boolean')
        
        self._arg             = arg
        self._shape           = arg.shape
        self._rank            = arg.rank
        self._dtype           = arg.dtype
        self._order           = arg.order
        self._precision       = arg.precision
        self._memory_location = memory_location
        self._is_async        = is_async
        super().__init__()
    
    @property
    def arg(self):
        return self._arg

    @property
    def memory_location(self):
        return self._memory_location

    @property
    def is_async(self):
        return self._is_async

class CudaThreadIdx(CudaInternalVar):
    __slots__ = ()
    pass
class CudaBlockDim(CudaInternalVar):
    __slots__ = ()
    pass
class CudaBlockIdx(CudaInternalVar):
    __slots__ = ()
    pass
class CudaGridDim(CudaInternalVar):
    __slots__ = ()
    pass

class CudaGrid(PyccelAstNode)               :
    """
    CudaGrid locate Thread In the GPU architecture Using CudaThreadIdx, CudaBlockDim, CudaBlockIdx
    To calculate the exact index of the thread automatically.

    Parameters
    ----------
    dim : NativeInteger
        Represent the dimension where we want to locate our thread.

    """
    __slots__ = ()
    _attribute_nodes = ()
    def __new__(cls, dim=0):
        if not isinstance(dim, LiteralInteger):
            raise TypeError("dimension need to be an integer")
        if dim not in (0, 1, 2):
            raise ValueError("dimension need to be 0, 1 or 2")
        expr = [PyccelAdd(PyccelMul(CudaBlockIdx(d), CudaBlockDim(d)), CudaThreadIdx(d))\
                for d in range(dim.python_value + 1)]
        if dim == 0:
            return expr[0]
        return PythonTuple(*expr)



cuda_funcs = {
    'array'             : PyccelFunctionDef('array'             , CudaArray),
    'copy'              : PyccelFunctionDef('copy'              , CudaCopy),
    'synchronize'       : PyccelFunctionDef('synchronize'       , CudaSynchronize),
    'threadIdx'         : PyccelFunctionDef('threadIdx'         , CudaThreadIdx),
    'blockDim'          : PyccelFunctionDef('blockDim'          , CudaBlockDim),
    'blockIdx'          : PyccelFunctionDef('blockIdx'          , CudaBlockIdx),
    'gridDim'           : PyccelFunctionDef('gridDim'           , CudaGridDim),
    'grid'              : PyccelFunctionDef('grid'              , CudaGrid)
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