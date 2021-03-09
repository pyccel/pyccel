#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

from .basic          import PyccelAstNode
from .builtins       import (PythonInt, PythonBool, PythonFloat, PythonTuple,
                             PythonComplex, PythonReal, PythonImag, PythonList)

from .core           import (ClassDef, FunctionDef,
                            process_shape, ValuedArgument)

from .datatypes      import (dtype_and_precision_registry as dtype_registry,
                             default_precision, datatype, NativeInteger,
                             NativeReal, NativeComplex, NativeBool, str_dtype,
                             NativeNumeric)

from .internals      import PyccelInternalFunction

from .literals       import LiteralInteger, LiteralFloat, LiteralComplex
from .literals       import LiteralTrue, LiteralFalse
from .literals       import Nil
from .mathext        import MathCeil
from .operators      import broadcast, PyccelMinus, PyccelDiv
from .variable       import (Variable, IndexedElement, Constant)

from .numpyext       import process_dtype


__all__ = (
    'CupyArrayClass',
    # ---
    'CupyEmpty',
    'CupyEmptyLike',
    'CupyFull',
    'CupyFullLike',
    'CupyOnes',
    'CupyOnesLike',
    'CupyZeros',
    'CupyZerosLike',
    'CupyArange'
)

#==============================================================================
class CupyNewArray(PyccelInternalFunction):
    """ Class from which all Cupy functions which imply a call to Allocate
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
class CupyArray(CupyNewArray):
    """
    Represents a call to  cupy.array for code generation.

    arg : list, tuple, PythonList

    """
    _attribute_nodes = ('_arg',)

    def __init__(self, arg, dtype=None, order='C'):

        if not isinstance(arg, (PythonTuple, PythonList, Variable)):
            raise TypeError('Unknown type of  %s.' % type(arg))

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
    _attribute_nodes = ('_start','_step','_stop')

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
            self._precision = max([i.precision for i in self.arg])
        else:
            self._dtype, self._precision = process_dtype(dtype)

        self._rank = 1
        self._shape = (MathCeil(PyccelDiv(PyccelMinus(self._stop, self._start), self._step)))
        self._shape = process_shape(self._shape)
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

def Shape(arg):
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

    def __init__(self, shape, fill_value, dtype=None, order='C'):

        # Convert shape to PythonTuple
        shape = process_shape(shape)

        # If there is no dtype, extract it from fill_value
        # TODO: must get dtype from an annotated node
        if not dtype:
            dtype = fill_value.dtype

        # Verify dtype and get precision
        dtype, precision = process_dtype(dtype)

        # Verify array ordering
        order = CupyNewArray._process_order(order)

        # Cast fill_value to correct type
        if fill_value and not isinstance(fill_value, Nil):
            cast_func = DtypePrecisionToCastFunction[dtype.name][precision]
            fill_value = cast_func(fill_value)
        self._shape = shape
        self._rank  = len(self._shape)
        self._dtype = dtype
        self._order = order
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
    def __init__(self, shape, dtype='float', order='C'):
        if not dtype:
            raise TypeError("Data type must be provided")

        super().__init__(shape, Nil(), dtype, order)

#==============================================================================
class CupyEmpty(CupyAutoFill):
    """ Represents a call to cupy.empty for code generation.
    """
    @property
    def fill_value(self):
        return None


#==============================================================================
class CupyZeros(CupyAutoFill):
    """ Represents a call to cupy.zeros for code generation.
    """
    @property
    def fill_value(self):
        dtype = self.dtype
        if isinstance(dtype, NativeInteger):
            value = LiteralInteger(0, precision = self.precision)
        elif isinstance(dtype, NativeReal):
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
    @property
    def fill_value(self):
        dtype = self.dtype
        if isinstance(dtype, NativeInteger):
            value = LiteralInteger(1, precision = self.precision)
        elif isinstance(dtype, NativeReal):
            value = LiteralFloat(1., precision = self.precision)
        elif isinstance(dtype, NativeComplex):
            value = LiteralComplex(1., 0., precision = self.precision)
        elif isinstance(dtype, NativeBool):
            value = LiteralTrue(precision = self.precision)
        else:
            raise TypeError('Unknown type')
        return value

#=======================================================================================
class CupyFullLike:
    """ Represents a call to cupy.full_like for code generation.
    """
    def __new__(cls, a, fill_value, dtype=None, order='K', subok=True, shape=None):

        # NOTE: we ignore 'subok' argument
        dtype = dtype or a.dtype
        order = a.order if str(order).strip('\'"') in ('K', 'A') else order
        shape = Shape(a) if shape is None else shape
        return CupyFull(shape, fill_value, dtype, order)

#=======================================================================================
class CupyEmptyLike:
    """ Represents a call to cupy.empty_like for code generation.
    """
    def __new__(cls, a, dtype=None, order='K', subok=True, shape=None):

        # NOTE: we ignore 'subok' argument
        dtype = dtype or a.dtype
        order = a.order if str(order).strip('\'"') in ('K', 'A') else order
        shape = Shape(a) if shape is None else shape

        return CupyEmpty(shape, dtype, order)

#=======================================================================================
class CupyOnesLike:
    """ Represents a call to cupy.ones_like for code generation.
    """
    def __new__(cls, a, dtype=None, order='K', subok=True, shape=None):

        # NOTE: we ignore 'subok' argument
        dtype = dtype or a.dtype
        order = a.order if str(order).strip('\'"') in ('K', 'A') else order
        shape = Shape(a) if shape is None else shape

        return CupyOnes(shape, dtype, order)

#=======================================================================================
class CupyZerosLike:
    """ Represents a call to cupy.zeros_like for code generation.
    """
    def __new__(cls, a, dtype=None, order='K', subok=True, shape=None):

        # NOTE: we ignore 'subok' argument
        dtype = dtype or a.dtype
        order = a.order if str(order).strip('\'"') in ('K', 'A') else order
        shape = Shape(a) if shape is None else shape

        return CupyZeros(shape, dtype, order)

#=======================================================================================

#==============================================================================
cupy_functions = {
    # ... array creation routines
    'full'      : CupyFull,
    'empty'     : CupyEmpty,
    'zeros'     : CupyZeros,
    'ones'      : CupyOnes,
    'full_like' : CupyFullLike,
    'empty_like': CupyEmptyLike,
    'zeros_like': CupyZerosLike,
    'ones_like' : CupyOnesLike,
    'array'     : CupyArray,
    'arange'    : CupyArange
}
