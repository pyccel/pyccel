from .basic          import Basic, PyccelAstNode
from .internals      import PyccelInternalFunction

from .core           import (ClassDef, FunctionDef,
                            process_shape, ValuedArgument)

from .literals       import LiteralInteger, LiteralFloat, LiteralComplex

from .datatypes      import (dtype_and_precision_registry as dtype_registry,
                             default_precision, datatype, NativeInteger,
                             NativeReal, NativeComplex, NativeBool, str_dtype,
                             NativeNumeric)

from .builtins       import (PythonInt, PythonBool, PythonFloat, PythonTuple,
                             PythonComplex, PythonReal, PythonImag, PythonList)

from .variable       import (Variable, IndexedElement, Constant)

from .numpyext       import process_dtype

#==============================================================================
__all__ = (
    'CudaArray',
    'CudaMalloc'
)


#------------------------------------------------------------------------------
#==============================================================================
class CudaMalloc(PyccelAstNode):
    """Represents a call to  cuda malloc for code generation.

    arg : list , tuple , PythonTuple, List, Variable
    """
    def __init__(self, size, alloct, dtype=NativeReal(), precision=4):
        self._size      = size
        self._shape     = (1,)
        self._alloct    = alloct
        self._rank      = 1
        self._dtype     = dtype
        self._precision = precision

    @property
    def size(self):
        return self._size
    @property
    def dtype(self):
        return self._dtype
    @property
    def precision(self):
        return self._precision

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
class CudaArray(CudaNewArray):
    """
    Represents a call to  Cuda.array for code generation.

    arg : list, tuple, PythonList

    """
    _attribute_nodes = ('_arg',)

    def __init__(self, arg, dtype=None, order='C'):

        if not isinstance(arg, (PythonTuple, PythonList, Variable)):
            raise TypeError('Unknown type of  %s.' % type(arg))

        # TODO: treat inhomogenous lists and tuples when they have mixed ordering
        if isinstance(arg, (PythonTuple, PythonList)) and not arg.is_homogeneous or \
            isinstance(arg, Variable) and not arg.is_ndarray and not arg.is_stack_array:
            raise TypeError('we only accept homogeneous arguments')

        # Verify dtype and get precision
        if dtype is None:
            dtype = arg.dtype
        dtype, prec = process_dtype(dtype)
        # ... Determine ordering
        if isinstance(order, ValuedArgument):
            order = order.value
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
    _attribute_nodes = ()
    pass

class CudaInternalVar(PyccelAstNode):
    _attribute_nodes = ('_dim',)

    def __init__(self, dim):
        print(dim, type(dim))
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
        super().__init__()
    
    @property
    def dim(self):
        return self._dim

class CudaThreadIdx(CudaInternalVar) : pass
class CudaBlockDim(CudaInternalVar)  : pass
class CudaBlockIdx(CudaInternalVar)  : pass
class CudaGridDim(CudaInternalVar)   : pass

cuda_functions = {
    'cudaMalloc'        : CudaMalloc,
    'deviceSynchronize' : CudaDeviceSynchronize,
    'array'             : CudaArray,
    'threadIdx'         : CudaThreadIdx,
    'blockDim'          : CudaBlockDim,
    'blockIdx'          : CudaBlockIdx,
    'gridDim'           : CudaGridDim
}