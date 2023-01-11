#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
""" Module containing objects from the cupy module understood by pyccel
"""

from functools import reduce
import operator

from pyccel.errors.errors import Errors
from pyccel.errors.messages import WRONG_LINSPACE_ENDPOINT, NON_LITERAL_KEEP_DIMS, NON_LITERAL_AXIS

from pyccel.utilities.stage import PyccelStage

from .basic          import PyccelAstNode
from .builtins       import (PythonInt, PythonBool, PythonFloat, PythonTuple,
                             PythonComplex, PythonReal, PythonImag, PythonList,
                             PythonType, PythonConjugate)

from .core           import Module, Import, PyccelFunctionDef, FunctionCall

from .datatypes      import (dtype_and_precision_registry as dtype_registry,
                             default_precision, datatype, NativeInteger,
                             NativeFloat, NativeComplex, NativeBool, str_dtype,
                             NativeNumeric)

from .internals      import PyccelInternalFunction, Slice, max_precision, get_final_precision
from .internals      import PyccelArraySize

from .literals       import LiteralInteger, LiteralFloat, LiteralComplex, LiteralString, convert_to_literal
from .literals       import LiteralTrue, LiteralFalse
from .literals       import Nil
from .mathext        import MathCeil
from .operators      import broadcast, PyccelMinus, PyccelDiv
from .variable       import (Variable, Constant, HomogeneousTupleVariable)
from .cudaext        import CudaNewArray, CudaArray
from .numpyext       import process_dtype, process_shape, DtypePrecisionToCastFunction

errors = Errors()
pyccel_stage = PyccelStage()

__all__ = (
    'CupyNewArray',
    'CupyArray',
    'CupyEmpty',
    'CupyEmptyLike',
    'CupyFull',
    'CupyFullLike',
    'CupyArange',
    'CupyArraySize',
    'CupyOnes',
    'CupyOnesLike',
    'Shape',
    'CupyZeros',
    'CupyZerosLike',
)

#==============================================================================
class CupyNewArray(CudaNewArray):
    """ Class from which all Cupy functions which imply a call to Allocate
    inherit
    """
    _memory_location = 'device'
    def __init__(self):
        super().__init__()

    @property
    def memory_location(self):
        """ Indicate if the array is allocated on the host, device or has a managed memory
        """
        return self._memory_location

#==============================================================================
class CupyArray(CupyNewArray):
    """
    Represents a call to  cupy.array for code generation.

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

        self._arg   = arg
        self._shape = shape
        self._rank  = rank
        self._dtype = dtype
        self._order = order
        self._precision = prec
        super().__init__()

    def __str__(self):
        return str(self.arg)

    @property
    def arg(self):
        return self._arg

#==============================================================================
class CupyArange(CupyNewArray):
    """
    Represents a call to  cupy.arange for code generation.

    Parameters
    ----------
    start : Numeric
        Start of interval, default value 0

    stop : Numeric
        End of interval

    step : Numeric
        Spacing between values, default value 1

    dtype : Datatype
        The type of the output array, if dtype is not given,
        infer the data type from the other input arguments.
    """
    __slots__ = ('_start','_step','_stop','_dtype','_precision','_shape')
    _attribute_nodes = ('_start','_step','_stop')
    _rank = 1
    _order = None
    name = 'arange'

    def __init__(self, start, stop = None, step = None, dtype = None):

        if stop is None:
            self._start = LiteralInteger(0)
            self._stop = start
        else:
            self._start = start
            self._stop = stop
        self._step = step if step is not None else LiteralInteger(1)

        if dtype is None:
            self._dtype = max([i.dtype for i in self.arg], key = NativeNumeric.index)
            self._precision = max_precision(self.arg, allow_native=False)
        else:
            self._dtype, self._precision = process_dtype(dtype)

        self._shape = (MathCeil(PyccelDiv(PyccelMinus(self._stop, self._start), self._step)))
        self._shape = process_shape(False, self._shape)
        super().__init__()

    @property
    def arg(self):
        return (self._start, self._stop, self._step)

    @property
    def start(self):
        return self._start

    @property
    def stop(self):
        return self._stop

    @property
    def step(self):
        return self._step

#==============================================================================

class Shape(PyccelInternalFunction):
    """ Represents a call to cupy.shape for code generation
    """
    __slots__ = ()
    name = 'shape'
    def __new__(cls, arg):
        if isinstance(arg.shape, PythonTuple):
            return arg.shape
        else:
            return PythonTuple(*arg.shape)

#==============================================================================
class CupyFull(CupyNewArray):
    """
    Represents a call to cupy.full for code generation.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.

    fill_value : scalar
        Fill value.

    dtype: str, DataType
        datatype for the constructed array
        The default, `None`, means `np.array(fill_value).dtype`.

    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory.

    """
    __slots__ = ('_fill_value','_dtype','_precision','_shape','_rank','_order')
    name = 'full'

    def __init__(self, shape, fill_value, dtype=None, order='C'):

        # Convert shape to PythonTuple
        shape = process_shape(False, shape)
        # If there is no dtype, extract it from fill_value
        # TODO: must get dtype from an annotated node
        if dtype is None:
            dtype = fill_value.dtype
            precision = get_final_precision(fill_value)
        else:
            dtype, precision = process_dtype(dtype)

        # Cast fill_value to correct type
        if fill_value:
            if fill_value.dtype != dtype or get_final_precision(fill_value) != precision:
                cast_func = DtypePrecisionToCastFunction[dtype.name][precision]
                fill_value = cast_func(fill_value)
        self._shape = shape
        self._rank  = len(self._shape)
        self._dtype = dtype
        self._order = CupyNewArray._process_order(self._rank, order)
        self._precision = precision

        super().__init__(fill_value)

    #--------------------------------------------------------------------------
    @property
    def fill_value(self):
        return self._args[0]

#==============================================================================
class CupyAutoFill(CupyFull):
    """ Abstract class for all classes which inherit from CupyFull but
        the fill_value is implicitly specified
    """
    __slots__ = ()
    def __init__(self, shape, dtype='float', order='C'):
        if not dtype:
            raise TypeError("Data type must be provided")
        super().__init__(shape, Nil(), dtype, order)

#==============================================================================
class CupyEmpty(CupyAutoFill):
    """ Represents a call to cupy.empty for code generation.
    """
    __slots__ = ()
    name = 'empty'

    def __init__(self, shape, dtype='float', order='C'):
        if dtype in NativeNumeric:
            precision = default_precision[str_dtype(dtype)]
            dtype = DtypePrecisionToCastFunction[dtype.name][precision]
        super().__init__(shape, dtype, order)
    @property
    def fill_value(self):
        return None


#==============================================================================
class CupyZeros(CupyAutoFill):
    """ Represents a call to cupy.zeros for code generation.
    """
    __slots__ = ()
    name = 'zeros'
    @property
    def fill_value(self):
        dtype = self.dtype
        if isinstance(dtype, NativeInteger):
            value = LiteralInteger(0, precision = self.precision)
        elif isinstance(dtype, NativeFloat):
            value = LiteralFloat(0, precision = self.precision)
        elif isinstance(dtype, NativeComplex):
            value = LiteralComplex(0., 0., precision = self.precision)
        elif isinstance(dtype, NativeBool):
            value = LiteralFalse(precision = self.precision)
        else:
            raise TypeError('Unknown type')
        return value

#==============================================================================
class CupyOnes(CupyAutoFill):
    """ Represents a call to cupy.ones for code generation.
    """
    __slots__ = ()
    name = 'ones'
    @property
    def fill_value(self):
        dtype = self.dtype
        if isinstance(dtype, NativeInteger):
            value = LiteralInteger(1, precision = self.precision)
        elif isinstance(dtype, NativeFloat):
            value = LiteralFloat(1., precision = self.precision)
        elif isinstance(dtype, NativeComplex):
            value = LiteralComplex(1., 0., precision = self.precision)
        elif isinstance(dtype, NativeBool):
            value = LiteralTrue(precision = self.precision)
        else:
            raise TypeError('Unknown type')
        return value

#=======================================================================================
class CupyFullLike(PyccelInternalFunction):
    """ Represents a call to cupy.full_like for code generation.
    """
    __slots__ = ()
    name = 'full_like'
    def __new__(cls, a, fill_value, dtype=None, order='K', subok=True, shape=None):

        # NOTE: we ignore 'subok' argument
        if dtype is None:
            dtype = DtypePrecisionToCastFunction[a.dtype.name][a.precision]
        order = a.order if str(order).strip('\'"') in ('K', 'A') else order
        shape = Shape(a) if shape is None else shape
        return CupyFull(shape, fill_value, dtype, order)

#=======================================================================================
class CupyEmptyLike(PyccelInternalFunction):
    """ Represents a call to cupy.empty_like for code generation.
    """
    __slots__ = ()
    name = 'empty_like'
    def __new__(cls, a, dtype=None, order='K', subok=True, shape=None):

        # NOTE: we ignore 'subok' argument
        if dtype is None:
            dtype = DtypePrecisionToCastFunction[a.dtype.name][a.precision]
        order = a.order if str(order).strip('\'"') in ('K', 'A') else order
        shape = Shape(a) if shape is None else shape

        return CupyEmpty(shape, dtype, order)

#=======================================================================================
class CupyOnesLike(PyccelInternalFunction):
    """ Represents a call to cupy.ones_like for code generation.
    """
    __slots__ = ()
    name = 'ones_like'
    def __new__(cls, a, dtype=None, order='K', subok=True, shape=None):

        # NOTE: we ignore 'subok' argument
        if dtype is None:
            dtype = DtypePrecisionToCastFunction[a.dtype.name][a.precision]
        order = a.order if str(order).strip('\'"') in ('K', 'A') else order
        shape = Shape(a) if shape is None else shape

        return CupyOnes(shape, dtype, order)

#=======================================================================================
class CupyZerosLike(PyccelInternalFunction):
    """ Represents a call to cupy.zeros_like for code generation.
    """
    __slots__ = ()
    name = 'zeros_like'
    def __new__(cls, a, dtype=None, order='K', subok=True, shape=None):

        # NOTE: we ignore 'subok' argument
        if dtype is None:
            dtype = DtypePrecisionToCastFunction[a.dtype.name][a.precision]
        order = a.order if str(order).strip('\'"') in ('K', 'A') else order
        shape = Shape(a) if shape is None else shape

        return CupyZeros(shape, dtype, order)

#=======================================================================================

class CupyArraySize(PyccelInternalFunction):
    """
    Class representing a call to the cupy size function which
    returns the shape of an object in a given dimension

    Parameters
    ==========
    arg   : PyccelAstNode
            A PyccelAstNode of unknown shape
    axis  : int
            The dimension along which the size is
            requested
    """
    __slots__ = ('_arg',)
    _attribute_nodes = ('_arg',)
    name   = 'size'
    _dtype = NativeInteger()
    _precision = -1
    _rank  = 0
    _shape = None
    _order = None

    def __new__(cls, a, axis = None):
        if axis is not None:
            return PyccelArraySize(a, axis)
        elif not isinstance(a, (list,
                                    tuple,
                                    PyccelAstNode)):
            raise TypeError('Unknown type of  %s.' % type(a))
        elif all(isinstance(s, LiteralInteger) for s in a.shape):
            return LiteralInteger(reduce(operator.mul, [s.python_value for s in a.shape]))
        else:
            return super().__new__(cls)

    def __init__(self, a, axis = None):
        self._arg   = a
        super().__init__(a)

    @property
    def arg(self):
        """ Object whose size is investigated
        """
        return self._arg

    def __str__(self):
        return 'Size({})'.format(str(self.arg))

#==============================================================================

cupy_funcs = {
    # ... array creation routines
    'full'      : PyccelFunctionDef('full'      , CupyFull),
    'empty'     : PyccelFunctionDef('empty'     , CupyEmpty),
    'zeros'     : PyccelFunctionDef('zeros'     , CupyZeros),
    'ones'      : PyccelFunctionDef('ones'      , CupyOnes),
    'full_like' : PyccelFunctionDef('full_like' , CupyFullLike),
    'empty_like': PyccelFunctionDef('empty_like', CupyEmptyLike),
    'zeros_like': PyccelFunctionDef('zeros_like', CupyZerosLike),
    'ones_like' : PyccelFunctionDef('ones_like' , CupyOnesLike),
    'array'     : PyccelFunctionDef('array'     , CupyArray),
    'arange'    : PyccelFunctionDef('arange'    , CupyArange),
    # ...
    'shape'     : PyccelFunctionDef('shape'     , Shape),
    'size'      : PyccelFunctionDef('size'      , CupyArraySize),
}

cuda_constants = {
}

cupy_mod = Module('cupy',
    variables = cuda_constants.values(),
    funcs = cupy_funcs.values())

#==============================================================================

cupy_target_swap = {
        cupy_funcs['full_like']  : cupy_funcs['full'],
        cupy_funcs['empty_like'] : cupy_funcs['empty'],
        cupy_funcs['zeros_like'] : cupy_funcs['zeros'],
        cupy_funcs['ones_like']  : cupy_funcs['ones']
    }
