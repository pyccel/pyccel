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

from .core           import process_shape, Module, Import, PyccelFunctionDef

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

from .numpyext       import process_dtype, NumpyNewArray

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
class CudaMemCopy():
    """Represents a call to  cuda malloc for code generation.

    arg : list , tuple , PythonTuple, List, Variable
    """
    def __init__(self, x, size):
        self._shape     = ()
        self._rank      = 0
        self._dtype     = x.dtype
        self._precision = x.precision

    @property
    def dest(self):
        return self._args[0]
    @property
    def src(self):
        return self._args[1]
    @property
    def size(self):
        return self._args[2]
    @property
    def copy_mode(self):
        return self._args[3]


#==============================================================================
class CudaNewArray(PyccelInternalFunction):
    """ Class from which all Cuda functions which imply a call to Allocate
    inherit
    """

    #--------------------------------------------------------------------------
    @staticmethod
    def _process_order(order):

        if not order:
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
    __slots__ = ('_arg','_dtype','_precision','_shape','_rank','_order')
    _attribute_nodes = ('_arg',)
    name = 'array'

    def __init__(self, arg, dtype=None, order='C'):

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
        dtype, prec = process_dtype(dtype)
        # ... Determine ordering
        order = str(order).strip("\'")

        if order not in ('K', 'A', 'C', 'F'):
            raise ValueError("Cannot recognize '{:s}' order".format(order))

        # TODO [YG, 18.02.2020]: set correct order based on input array
        if order in ('K', 'A'):
            order = 'C'
        # ...
        self._arg   = arg
        self._shape = process_shape(arg.shape)
        self._rank  = len(self._shape)
        self._dtype = dtype
        self._order = order
        self._precision = prec
        super().__init__()

    def __str__(self):
        return str(self.arg)

    @property
    def arg(self):
        return self._arg

class CudaDeviceSynchronize(PyccelAstNode):
    "Represents a call to  Cuda.deviceSynchronize for code generation."

    def __init__(self):
        #...
        self._shape     = ()
        self._rank      = 0
        self._dtype     = NativeInteger()
        self._precision = 0
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
        self._shape     = ()
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