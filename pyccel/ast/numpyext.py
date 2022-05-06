#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
""" Module containing objects from the numpy module understood by pyccel
"""

import numpy

from pyccel.errors.errors import Errors
from pyccel.errors.messages import WRONG_LINSPACE_ENDPOINT

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

from .literals       import LiteralInteger, LiteralFloat, LiteralComplex, convert_to_literal
from .literals       import LiteralTrue, LiteralFalse
from .literals       import Nil
from .mathext        import MathCeil
from .operators      import broadcast, PyccelMinus, PyccelDiv
from .variable       import (Variable, Constant, HomogeneousTupleVariable)

errors = Errors()
pyccel_stage = PyccelStage()

__all__ = (
    'NumpyAbs',
    'NumpyFloor',
    # ---
    'NumpySqrt',
    'NumpySin',
    'NumpyCos',
    'NumpyExp',
    'NumpyLog',
    'NumpyTan',
    'NumpyArcsin',
    'NumpyArccos',
    'NumpyArctan',
    'NumpyArctan2',
    'NumpySinh',
    'NumpyCosh',
    'NumpyTanh',
    'NumpyArcsinh',
    'NumpyArccosh',
    'NumpyArctanh',
    # ---
    'NumpyEmpty',
    'NumpyEmptyLike',
    'NumpyFloat',
    'NumpyComplex',
    'NumpyComplex64',
    'NumpyComplex128',
    'NumpyFloat32',
    'NumpyFloat64',
    'NumpyFull',
    'NumpyFullLike',
    'NumpyImag',
    'NumpyBool',
    'NumpyInt',
    'NumpyInt8',
    'NumpyInt16',
    'NumpyInt32',
    'NumpyInt64',
    'NumpyLinspace',
    'NumpyMatmul',
    'NumpyAmax',
    'NumpyAmin',
    'NumpyMod',
    'NumpyNorm',
    'NumpySum',
    'NumpyOnes',
    'NumpyOnesLike',
    'NumpyProduct',
    'NumpyRand',
    'NumpyRandint',
    'NumpyReal',
    'Shape',
    'NumpyWhere',
    'NumpyZeros',
    'NumpyZerosLike',
    'NumpyArange'
)

#=======================================================================================
class NumpyFloat(PythonFloat):
    """ Represents a call to numpy.float() function.
    """
    __slots__ = ('_rank','_shape','_order')
    name = 'float'
    def __init__(self, arg):
        self._shape = arg.shape
        self._rank  = arg.rank
        self._order = arg.order
        super().__init__(arg)

class NumpyFloat32(NumpyFloat):
    """ Represents a call to numpy.float32() function.
    """
    __slots__ = ()
    _precision = dtype_registry['float32'][1]
    name = 'float32'

class NumpyFloat64(NumpyFloat):
    """ Represents a call to numpy.float64() function.
    """
    __slots__ = ()
    _precision = dtype_registry['float64'][1]
    name = 'float64'

#=======================================================================================
class NumpyBool(PythonBool):
    """ Represents a call to numpy.bool() function.
    """
    __slots__ = ('_shape','_rank','_order')
    name = 'bool'
    def __init__(self, arg):
        self._shape = arg.shape
        self._rank  = arg.rank
        self._order = arg.order
        super().__init__(arg)

#=======================================================================================
# TODO [YG, 13.03.2020]: handle case where base != 10
class NumpyInt(PythonInt):
    """ Represents a call to numpy.int() function.
    """
    __slots__ = ('_shape','_rank','_order')
    name = 'int'
    def __init__(self, arg=None, base=10):
        self._shape = arg.shape
        self._rank  = arg.rank
        self._order = arg.order
        super().__init__(arg)

class NumpyInt8(NumpyInt):
    """ Represents a call to numpy.int8() function.
    """
    _precision = dtype_registry['int8'][1]
    name = 'int8'

class NumpyInt16(NumpyInt):
    """ Represents a call to numpy.int16() function.
    """
    _precision = dtype_registry['int16'][1]
    name = 'int16'

class NumpyInt32(NumpyInt):
    """ Represents a call to numpy.int32() function.
    """
    __slots__ = ()
    _precision = dtype_registry['int32'][1]
    name = 'int32'

class NumpyInt64(NumpyInt):
    """ Represents a call to numpy.int64() function.
    """
    __slots__ = ()
    _precision = dtype_registry['int64'][1]
    name = 'int64'

#==============================================================================
class NumpyReal(PythonReal):
    """Represents a call to  numpy.real for code generation.

    > a = 1+2j
    > np.real(a)
    1.0
    """
    __slots__ = ('_precision','_rank','_shape','_order')
    name = 'real'
    def __new__(cls, arg):
        if isinstance(arg.dtype, NativeBool):
            return NumpyInt(arg)
        else:
            return super().__new__(cls, arg)

    def __init__(self, arg):
        super().__init__(arg)
        self._precision = arg.precision
        self._order = arg.order
        self._shape = process_shape(self.internal_var.shape)
        self._rank  = len(self._shape)

    @property
    def is_elemental(self):
        """ Indicates whether the function should be
        called elementwise for an array argument
        """
        return True

#==============================================================================

class NumpyImag(PythonImag):
    """Represents a call to  numpy.imag for code generation.

    > a = 1+2j
    > np.imag(a)
    2.0
    """
    __slots__ = ('_precision','_rank','_shape','_order')
    name = 'imag'
    def __new__(cls, arg):
        if not isinstance(arg.dtype, NativeComplex):
            dtype=NativeInteger() if isinstance(arg.dtype, NativeBool) else arg.dtype
            if arg.rank == 0:
                return convert_to_literal(0, dtype, arg.precision)
            return NumpyZeros(arg.shape, dtype=dtype)
        return super().__new__(cls, arg)

    def __init__(self, arg):
        super().__init__(arg)
        self._precision = arg.precision
        self._order = arg.order
        self._shape = self.internal_var.shape
        self._rank  = len(self._shape)

    @property
    def is_elemental(self):
        """ Indicates whether the function should be
        called elementwise for an array argument
        """
        return True

#=======================================================================================
class NumpyComplex(PythonComplex):
    """ Represents a call to numpy.complex() function.
    """
    _real_cast = NumpyReal
    _imag_cast = NumpyImag
    __slots__ = ('_rank','_shape','_order')
    name = 'complex'
    def __init__(self, arg0, arg1 = None):
        if arg1 is not None:
            raise NotImplementedError("Use builtin complex function not deprecated np.complex")
        self._shape = arg0.shape
        self._rank  = arg0.rank
        self._order = arg0.order
        super().__init__(arg0)

class NumpyComplex64(NumpyComplex):
    """ Represents a call to numpy.complex64() function.
    """
    __slots__ = ()
    _precision = dtype_registry['complex64'][1]
    name = 'complex64'

class NumpyComplex128(NumpyComplex):
    """ Represents a call to numpy.complex128() function.
    """
    __slots__ = ()
    _precision = dtype_registry['complex128'][1]
    name = 'complex128'

DtypePrecisionToCastFunction = {
    'Int' : {
       -1 : PythonInt,
        1 : NumpyInt8,
        2 : NumpyInt16,
        4 : NumpyInt32,
        8 : NumpyInt64},
    'Float' : {
       -1 : PythonFloat,
        4 : NumpyFloat32,
        8 : NumpyFloat64},
    'Complex' : {
       -1 : PythonComplex,
        4 : NumpyComplex64,
        8 : NumpyComplex,
        16 : NumpyComplex128,},
    'Bool':  {
       -1 : PythonBool,
        4 : NumpyBool}
}

#==============================================================================

def process_dtype(dtype):
    if isinstance(dtype, PythonType):
        return dtype.dtype, get_final_precision(dtype)
    if isinstance(dtype, PyccelFunctionDef):
        dtype = dtype.cls_name

    if dtype  in (PythonInt, PythonFloat, PythonComplex, PythonBool):
        # remove python prefix from dtype.name len("python") = 6
        dtype = dtype.__name__.lower()[6:]
    elif dtype  in (NumpyInt, NumpyInt8, NumpyInt16, NumpyInt32, NumpyInt64, NumpyComplex, NumpyFloat,
				  NumpyComplex128, NumpyComplex64, NumpyFloat64, NumpyFloat32):
        # remove numpy prefix from dtype.name len("numpy") = 5
        dtype = dtype.__name__.lower()[5:]
    else:
        dtype            = str(dtype).replace('\'', '').lower()
    dtype, precision = dtype_registry[dtype]
    if precision == -1:
        precision        = default_precision[dtype]
    dtype            = datatype(dtype)

    return dtype, precision

#==============================================================================
class NumpyNewArray(PyccelInternalFunction):
    """ Class from which all numpy functions which imply a call to Allocate
    inherit
    """
    __slots__ = ()

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
# TODO [YG, 18.02.2020]: accept Numpy array argument
# TODO [YG, 18.02.2020]: use order='K' as default, like in numpy.array
# TODO [YG, 22.05.2020]: move dtype & prec processing to __init__
class NumpyArray(NumpyNewArray):
    """
    Represents a call to  numpy.array for code generation.

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

#==============================================================================
class NumpyArange(NumpyNewArray):
    """
    Represents a call to  numpy.arange for code generation.

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
class NumpySum(PyccelInternalFunction):
    """Represents a call to  numpy.sum for code generation.

    arg : list , tuple , PythonTuple, PythonList, Variable
    """
    __slots__ = ('_dtype','_precision')
    name = 'sum'
    _rank  = 0
    _shape = ()
    _order = None

    def __init__(self, arg):
        if not isinstance(arg, PyccelAstNode):
            raise TypeError('Unknown type of  %s.' % type(arg))
        super().__init__(arg)
        self._dtype = arg.dtype
        self._precision = get_final_precision(arg)

    @property
    def arg(self):
        return self._args[0]

#==============================================================================
class NumpyProduct(PyccelInternalFunction):
    """Represents a call to  numpy.prod for code generation.

    arg : list , tuple , PythonTuple, PythonList, Variable
    """
    __slots__ = ('_arg','_dtype','_precision')
    name = 'product'
    _rank  = 0
    _shape = ()
    _order = None

    def __init__(self, arg):
        if not isinstance(arg, PyccelAstNode):
            raise TypeError('Unknown type of  %s.' % type(arg))
        super().__init__(arg)
        self._arg = PythonList(arg) if arg.rank == 0 else self._args[0]
        self._arg = NumpyInt(self._arg) if (isinstance(arg.dtype, NativeBool) or \
                    (isinstance(arg.dtype, NativeInteger) and get_final_precision(self._arg) < default_precision['int']))\
                    else self._arg
        self._dtype = self._arg.dtype
        self._precision = get_final_precision(self._arg)

    @property
    def arg(self):
        return self._arg


#==============================================================================
class NumpyMatmul(PyccelInternalFunction):
    """Represents a call to numpy.matmul for code generation.
    arg : list , tuple , PythonTuple, PythonList, Variable
    """
    __slots__ = ('_dtype','_precision','_shape','_rank','_order')
    name = 'matmul'

    def __init__(self, a ,b):
        super().__init__(a, b)
        if pyccel_stage == 'syntactic':
            return

        if not isinstance(a, PyccelAstNode):
            raise TypeError('Unknown type of  %s.' % type(a))
        if not isinstance(b, PyccelAstNode):
            raise TypeError('Unknown type of  %s.' % type(a))

        args      = (a, b)
        integers  = [e for e in args if e.dtype is NativeInteger()]
        booleans  = [e for e in args if e.dtype is NativeBool()]
        floats    = [e for e in args if e.dtype is NativeFloat()]
        complexs  = [e for e in args if e.dtype is NativeComplex()]

        if complexs:
            self._dtype     = NativeComplex()
            self._precision = max_precision(complexs, allow_native = False)
        elif floats:
            self._dtype     = NativeFloat()
            self._precision = max_precision(floats, allow_native = False)
        elif integers:
            self._dtype     = NativeInteger()
            self._precision = max_precision(integers, allow_native = False)
        elif booleans:
            self._dtype     = NativeBool()
            self._precision = max_precision(booleans, allow_native = False)
        else:
            raise TypeError('cannot determine the type of {}'.format(self))

        if not (a.shape is None or b.shape is None):

            m = 1 if a.rank < 2 else a.shape[0]
            n = 1 if b.rank < 2 else b.shape[1]
            self._shape = (m, n)

        if a.rank == 1 and b.rank == 1:
            self._rank = 0
            self._shape = ()
        elif a.rank == 1 or b.rank == 1:
            self._rank = 1
            self._shape = (b.shape[1] if a.rank == 1 else a.shape[0],)
        else:
            self._rank = 2


        if a.order == b.order:
            self._order = a.order
        else:
            self._order = 'C'

    @property
    def a(self):
        return self._args[0]

    @property
    def b(self):
        return self._args[1]


#==============================================================================

class Shape(PyccelInternalFunction):
    """ Represents a call to numpy.shape for code generation
    """
    __slots__ = ()
    name = 'shape'
    def __new__(cls, arg):
        if isinstance(arg.shape, PythonTuple):
            return arg.shape
        else:
            return PythonTuple(*arg.shape)

#==============================================================================
class NumpyLinspace(NumpyNewArray):

    """
    Represents numpy.linspace which returns num evenly spaced samples, calculated over the interval [start, stop].

    Parameters
      ----------
      start           : list , tuple , PythonTuple, PythonList, Variable, Literals
                        Represents the starting value of the sequence.
      stop            : list , tuple , PythonTuple, PythonList, Variable, Literals
                        Represents the ending value of the sequence (if endpoint is set to False).
      num             : int, optional
                        Number of samples to generate. Default is 50. Must be non-negative.
      endpoint        : bool, optional
                        If True, stop is the last sample. Otherwise, it is not included. Default is True.
      dtype           : str, DataType
                        The type of the output array. If dtype is not given, the data type is calculated
                        from start and stop, the calculated dtype will never be an integer.
    """

    __slots__ = ('_dtype','_precision','_index','_start','_stop','_num','_endpoint','_shape', '_rank','_ind','_step','_py_argument')
    _attribute_nodes = ('_start', '_stop', '_index', '_step', '_num', '_endpoint', '_ind')
    name = 'linspace'
    _order     = 'C'

    def __init__(self, start, stop, num=None, endpoint=True, dtype=None):

        if not num:
            num = LiteralInteger(50)

        if num.rank != 0 or not isinstance(num.dtype, NativeInteger):
            raise TypeError('Expecting positive integer num argument.')

        if any(not isinstance(arg, PyccelAstNode) for arg in (start, stop, num)):
            raise TypeError('Expecting valid args.')

        if dtype:
            self._dtype, self._precision = process_dtype(dtype)
        else:
            args      = (start, stop)
            integers  = [e for e in args if e.dtype is NativeInteger()]
            floats    = [e for e in args if e.dtype is NativeFloat()]
            complexs  = [e for e in args if e.dtype is NativeComplex()]

            if complexs:
                self._dtype     = NativeComplex()
                self._precision = max_precision(complexs, allow_native = False)
            elif floats:
                self._dtype     = NativeFloat()
                self._precision = max_precision(floats, allow_native = False)
            elif integers:
                self._dtype     = NativeFloat()
                self._precision = default_precision['float']
            else:
                raise TypeError('cannot determine the type of {}'.format(self))

        self._index = Variable('int', 'linspace_index')
        self._start = start
        self._stop  = stop
        self._num  = num
        if endpoint is True:
            self._endpoint = LiteralTrue()
        elif endpoint is False:
            self._endpoint = LiteralFalse()
        else:
            if not isinstance(endpoint.dtype, NativeBool):
                errors.report(WRONG_LINSPACE_ENDPOINT, symbol=endpoint, severity="fatal")
            self._endpoint = endpoint

        shape = broadcast(self._start.shape, self._stop.shape)
        self._shape = (self._num,) + shape
        self._rank  = len(self._shape)
        self._ind = None

        if isinstance(self.endpoint, LiteralFalse):
            self._step = PyccelDiv(PyccelMinus(self._stop, self._start), self.num)
        elif isinstance(self.endpoint, LiteralTrue):
            self._step = PyccelDiv(PyccelMinus(self._stop, self._start), PyccelMinus(self.num, LiteralInteger(1), simplify=True))
        else:
            self._step = PyccelDiv(PyccelMinus(self.stop, self.start), PyccelMinus(self.num, PythonInt(self.endpoint)))

        super().__init__()

    @property
    def dtype(self):
        return self._dtype

    @property
    def endpoint(self):
        """Indicates if the stop must be included or not."""
        return self._endpoint

    @property
    def start(self):
        """Represent the starting value of the sequence."""
        return self._start

    @property
    def stop(self):
        """Represent the end value of the sequence, if the endpoint is False the stop will not be included."""
        return self._stop

    @property
    def num(self):
        """Represent the number of generated elements by the linspace function."""
        return self._num

    @property
    def index(self):
        """Used in the fortran codegen when there is no for loop created."""
        return self._index

    @property
    def rank(self):
        return self._rank

    @property
    def step(self):
        """Represent size of spacing between generated elements."""
        return self._step

    @property
    def ind(self):
        """Used to store the index generated by the created for loop and needed by linspace function."""
        return self._ind

    @ind.setter
    def ind(self, value):
        assert self._ind is None
        value.set_current_user_node(self)
        self._ind = value

    @property
    def is_elemental(self):
        return True

#==============================================================================
class NumpyWhere(PyccelInternalFunction):
    """ Represents a call to  numpy.where """
    __slots__ = ()
    name = 'where'

    def __init__(self, mask):
        super().__init__(mask)


    @property
    def mask(self):
        return self._args[0]

    @property
    def index(self):
        ind = Variable('int','ind1')

        return ind

 #==============================================================================
class NumpyRand(PyccelInternalFunction):

    """
      Represents a call to  numpy.random.random or numpy.random.rand for code generation.

    """
    __slots__ = ('_shape','_rank')
    name = 'rand'
    _dtype = NativeFloat()
    _precision = default_precision['float']
    _order = 'C'

    def __init__(self, *args):
        super().__init__(*args)
        self._shape = args
        self._rank  = len(self.shape)

#==============================================================================
class NumpyRandint(PyccelInternalFunction):

    """
      Represents a call to  numpy.random.random or numpy.random.rand for code generation.

    """
    __slots__ = ('_rand','_low','_high','_shape','_rank')
    name = 'randint'
    _dtype     = NativeInteger()
    _precision = -1
    _order     = 'C'
    _attribute_nodes = ('_low', '_high')

    def __init__(self, low, high = None, size = None):
        if size is None:
            size = ()
        if not hasattr(size,'__iter__'):
            size = (size,)

        if high is None:
            high = low
            low  = None

        self._shape   = size
        self._rank    = len(self.shape)
        self._rand    = NumpyRand(*size)
        self._low     = low
        self._high    = high
        super().__init__()

    @property
    def rand_expr(self):
        return self._rand

    @property
    def high(self):
        """ return high property of NumpyRandint"""
        return self._high

    @property
    def low(self):
        """ return low property of NumpyRandint"""
        return self._low

#==============================================================================
class NumpyFull(NumpyNewArray):
    """
    Represents a call to numpy.full for code generation.

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
        shape = process_shape(shape)

        # If there is no dtype, extract it from fill_value
        # TODO: must get dtype from an annotated node
        if not dtype:
            dtype = fill_value.dtype

        # Verify dtype and get precision
        dtype, precision = process_dtype(dtype)

        # Verify array ordering
        order = NumpyNewArray._process_order(order)

        # Cast fill_value to correct type
        if fill_value:
            if fill_value.dtype != dtype or get_final_precision(fill_value) != precision:
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
class NumpyAutoFill(NumpyFull):
    """ Abstract class for all classes which inherit from NumpyFull but
        the fill_value is implicitly specified
    """
    __slots__ = ()
    def __init__(self, shape, dtype='float', order='C'):
        if not dtype:
            raise TypeError("Data type must be provided")

        super().__init__(shape, Nil(), dtype, order)

#==============================================================================
class NumpyEmpty(NumpyAutoFill):
    """ Represents a call to numpy.empty for code generation.
    """
    __slots__ = ()
    name = 'empty'
    @property
    def fill_value(self):
        return None


#==============================================================================
class NumpyZeros(NumpyAutoFill):
    """ Represents a call to numpy.zeros for code generation.
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
class NumpyOnes(NumpyAutoFill):
    """ Represents a call to numpy.ones for code generation.
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
class NumpyFullLike(PyccelInternalFunction):
    """ Represents a call to numpy.full_like for code generation.
    """
    __slots__ = ()
    name = 'full_like'
    def __new__(cls, a, fill_value, dtype=None, order='K', subok=True, shape=None):

        # NOTE: we ignore 'subok' argument
        dtype = dtype or a.dtype
        order = a.order if str(order).strip('\'"') in ('K', 'A') else order
        shape = Shape(a) if shape is None else shape
        return NumpyFull(shape, fill_value, dtype, order)

#=======================================================================================
class NumpyEmptyLike(PyccelInternalFunction):
    """ Represents a call to numpy.empty_like for code generation.
    """
    __slots__ = ()
    name = 'empty_like'
    def __new__(cls, a, dtype=None, order='K', subok=True, shape=None):

        # NOTE: we ignore 'subok' argument
        dtype = dtype or a.dtype
        order = a.order if str(order).strip('\'"') in ('K', 'A') else order
        shape = Shape(a) if shape is None else shape

        return NumpyEmpty(shape, dtype, order)

#=======================================================================================
class NumpyOnesLike(PyccelInternalFunction):
    """ Represents a call to numpy.ones_like for code generation.
    """
    __slots__ = ()
    name = 'ones_like'
    def __new__(cls, a, dtype=None, order='K', subok=True, shape=None):

        # NOTE: we ignore 'subok' argument
        dtype = dtype or a.dtype
        order = a.order if str(order).strip('\'"') in ('K', 'A') else order
        shape = Shape(a) if shape is None else shape

        return NumpyOnes(shape, dtype, order)

#=======================================================================================
class NumpyZerosLike(PyccelInternalFunction):
    """ Represents a call to numpy.zeros_like for code generation.
    """
    __slots__ = ()
    name = 'zeros_like'
    def __new__(cls, a, dtype=None, order='K', subok=True, shape=None):

        # NOTE: we ignore 'subok' argument
        dtype = dtype or a.dtype
        order = a.order if str(order).strip('\'"') in ('K', 'A') else order
        shape = Shape(a) if shape is None else shape

        return NumpyZeros(shape, dtype, order)

#=======================================================================================

class NumpyNorm(PyccelInternalFunction):
    """ Represents call to numpy.norm"""
    __slots__ = ('_shape','_rank','_order','_arg','_precision')
    name = 'norm'
    _dtype = NativeFloat()

    def __init__(self, arg, axis=None):
        super().__init__(arg, axis)
        if not isinstance(arg.dtype, (NativeComplex, NativeFloat)):
            arg = NumpyFloat(arg)
        self._arg = PythonList(arg) if arg.rank == 0 else arg
        self._precision = get_final_precision(arg)
        if self.axis is not None:
            sh = list(arg.shape)
            del sh[self.axis]
            self._shape = tuple(sh)
            self._order = arg.order
        else:
            self._shape = ()
            self._order = None
        self._rank = len(self._shape)
        self._order = arg.order

    @property
    def arg(self):
        return self._arg

    @property
    def python_arg(self):
        """numpy.norm argument without casting.
        the actual arg property contains casting methods for C/Fortran,
        which is not necessary for a Python code, and the casting makes Python language tests fail.
        """
        return self._args[0]

    @property
    def axis(self):
        """
        Mimic the behavior of axis argument of numpy.norm in python,
        and dim argument of Norm2 in Fortran.
        """
        return self._args[1]

#====================================================


#==============================================================================
# Numpy universal functions
# https://numpy.org/doc/stable/reference/ufuncs.html#available-ufuncs
#==============================================================================
class NumpyUfuncBase(PyccelInternalFunction):
    """Base class for Numpy's universal functions."""
    __slots__ = ('_dtype','_precision','_shape','_rank','_order')
    @property
    def is_elemental(self):
        return True

#------------------------------------------------------------------------------
class NumpyUfuncUnary(NumpyUfuncBase):
    """Numpy's universal function with one argument.
    """
    __slots__ = ()
    def __init__(self, x):
        self._set_dtype_precision(x)
        self._set_shape_rank(x)
        self._set_order(x)
        super().__init__(x)

    def _set_shape_rank(self, x):
        self._shape      = x.shape
        self._rank       = x.rank

    def _set_dtype_precision(self, x):
        self._dtype      = x.dtype if x.dtype is NativeComplex() else NativeFloat()
        self._precision  = default_precision[str_dtype(self._dtype)]

    def _set_order(self, x):
        self._order      = x.order

#------------------------------------------------------------------------------
class NumpyUfuncBinary(NumpyUfuncBase):
    """Numpy's universal function with two arguments.
    """
    __slots__ = ()
    # TODO: apply Numpy's broadcasting rules to get shape/rank of output
    def __init__(self, x1, x2):
        super().__init__(x1, x2)
        self._set_dtype_precision(x1, x2)
        self._set_shape_rank(x1, x2)
        self._set_order(x1, x2)

    def _set_shape_rank(self, x1, x2):
        self._shape     = x1.shape  # TODO ^^
        self._rank      = x1.rank   # TODO ^^

    def _set_dtype_precision(self, x1, x2):
        self._dtype     = NativeFloat()
        self._precision = default_precision['float']

    def _set_order(self, x1, x2):
        if x1.order == x2.order:
            self._order = x1.order
        else:
            self._order = 'C'

#------------------------------------------------------------------------------
# Math operations
#------------------------------------------------------------------------------
#class NumpyAbsolute(NumpyUfuncUnary): __slots__ = ()
class NumpyFabs    (NumpyUfuncUnary):
    """Represent a call to the fabs function in the Numpy library"""
    __slots__ = ()
    name = 'fabs'
class NumpyExp     (NumpyUfuncUnary):
    """Represent a call to the exp function in the Numpy library"""
    __slots__ = ()
    name = 'exp'
class NumpyLog     (NumpyUfuncUnary):
    """Represent a call to the log function in the Numpy library"""
    __slots__ = ()
    name = 'log'
class NumpySqrt    (NumpyUfuncUnary):
    """Represent a call to the sqrt function in the Numpy library"""
    __slots__ = ()
    name = 'sqrt'

#------------------------------------------------------------------------------
# Trigonometric functions
#------------------------------------------------------------------------------
class NumpySin    (NumpyUfuncUnary):
    """Represent a call to the sin function in the Numpy library"""
    __slots__ = ()
    name = 'sin'
class NumpyCos    (NumpyUfuncUnary):
    """Represent a call to the cos function in the Numpy library"""
    __slots__ = ()
    name = 'cos'
class NumpyTan    (NumpyUfuncUnary):
    """Represent a call to the tan function in the Numpy library"""
    __slots__ = ()
    name = 'tan'
class NumpyArcsin (NumpyUfuncUnary):
    """Represent a call to the arcsin function in the Numpy library"""
    __slots__ = ()
    name = 'arcsin'
class NumpyArccos (NumpyUfuncUnary):
    """Represent a call to the arccos function in the Numpy library"""
    __slots__ = ()
    name = 'arccos'
class NumpyArctan (NumpyUfuncUnary):
    """Represent a call to the arctan function in the Numpy library"""
    __slots__ = ()
    name = 'arctan'
class NumpyArctan2(NumpyUfuncBinary):
    """Represent a call to the arctan2 function in the Numpy library"""
    __slots__ = ()
    name = 'arctan2'
class NumpyHypot  (NumpyUfuncBinary):
    """Represent a call to the hypot function in the Numpy library"""
    __slots__ = ()
    name = 'hypot'
class NumpySinh   (NumpyUfuncUnary):
    """Represent a call to the sinh function in the Numpy library"""
    __slots__ = ()
    name = 'sinh'
class NumpyCosh   (NumpyUfuncUnary):
    """Represent a call to the cosh function in the Numpy library"""
    __slots__ = ()
    name = 'cosh'
class NumpyTanh   (NumpyUfuncUnary):
    """Represent a call to the tanh function in the Numpy library"""
    __slots__ = ()
    name = 'tanh'
class NumpyArcsinh(NumpyUfuncUnary):
    """Represent a call to the arcsinh function in the Numpy library"""
    __slots__ = ()
    name = 'arcsinh'
class NumpyArccosh(NumpyUfuncUnary):
    """Represent a call to the arccosh function in the Numpy library"""
    __slots__ = ()
    name = 'arccosh'
class NumpyArctanh(NumpyUfuncUnary):
    """Represent a call to the arctanh function in the Numpy library"""
    __slots__ = ()
    name = 'arctanh'
#class NumpyDeg2rad(NumpyUfuncUnary):
#    """Represent a call to the numpydeg2rad function in the Numpy library"""
#    __slots__ = ()
#    name = 'deg2rad'
#class NumpyRad2deg(NumpyUfuncUnary):
#    """Represent a call to the numpyrad2deg function in the Numpy library"""
#     __slots__ = ()
#     name = 'rad2deg'

#=======================================================================================

class NumpyAbs(NumpyUfuncUnary):
    """Represent a call to the abs function in the Numpy library"""
    __slots__ = ()
    name = 'abs'
    def _set_dtype_precision(self, x):
        self._dtype     = NativeInteger() if x.dtype is NativeInteger() else NativeFloat()
        self._precision = default_precision[str_dtype(self._dtype)]

class NumpyFloor(NumpyUfuncUnary):
    """Represent a call to the floor function in the Numpy library"""
    __slots__ = ()
    name = 'floor'
    def _set_dtype_precision(self, x):
        self._dtype     = NativeFloat()
        self._precision = default_precision[str_dtype(self._dtype)]

class NumpyMod(NumpyUfuncBinary):
    """Represent a call to the mod function in the Numpy library"""
    __slots__ = ()
    name = 'mod'

    def __init__(self, x1, x2):
        super().__init__(x1, x2)
        x1 = NumpyInt(x1) if isinstance(x1.dtype, NativeBool) else x1
        x2 = NumpyInt(x2) if isinstance(x2.dtype, NativeBool) else x2
        self._args = (x1, x2)

    def _set_shape_rank(self, x1, x2):
        args   = (x1, x2)
        shapes = [a.shape for a in args]

        if all(sh is not None for sh in shapes):
            if len(args) == 1:
                shape = args[0].shape
            else:
                shape = broadcast(args[0].shape, args[1].shape)

                for a in args[2:]:
                    shape = broadcast(shape, a.shape)

            self._shape = shape
            self._rank  = len(shape)
        else:
            self._rank = max(a.rank for a in args)

    def _set_dtype_precision(self, x1, x2):
        args      = (x1, x2)
        integers  = [a for a in args if a.dtype is NativeInteger() or a.dtype is NativeBool()]
        floats    = [a for a in args if a.dtype is NativeFloat()]
        others    = [a for a in args if a not in integers+floats]

        if others:
            raise TypeError('{} not supported'.format(others[0].dtype))

        if floats:
            self._dtype     = NativeFloat()
            self._precision = max_precision(floats, allow_native = False)
        elif integers:
            self._dtype     = NativeInteger()
            integers  = [a for a in args if a.dtype is NativeInteger()]
            if integers:
                self._precision = max_precision(integers, NativeInteger(), allow_native = False)
            else:
                self._precision = 1
        else:
            raise TypeError('cannot determine the type of {}'.format(self))

class NumpyAmin(NumpyUfuncUnary):
    """Represent a call to the amin function in the Numpy library"""
    __slots__ = ()
    name = 'amin'
    def _set_shape_rank(self, x):
        self._shape     = ()
        self._rank      = 0

    def _set_dtype_precision(self, x):
        self._dtype     = x.dtype
        self._precision = get_final_precision(x)

    @property
    def is_elemental(self):
        return False

class NumpyAmax(NumpyUfuncUnary):
    """Represent a call to the amax function in the Numpy library"""
    __slots__ = ()
    name = 'amax'
    def _set_shape_rank(self, x):
        self._shape     = ()
        self._rank      = 0

    def _set_dtype_precision(self, x):
        self._dtype     = x.dtype
        self._precision = get_final_precision(x)

    @property
    def is_elemental(self):
        return False

class NumpyTranspose(NumpyUfuncUnary):
    """Represents a call to the transpose function in the Numpy library"""
    __slots__ = ()
    name = 'transpose'

    def __new__(cls, x, *axes):
        if x.rank<2:
            return x
        else:
            return super().__new__(cls)

    def __init__(self, x, *axes):
        if len(axes)!=0:
            raise NotImplementedError("The axes argument of the transpose function is not yet implemented")
        super().__init__(x)

    @property
    def internal_var(self):
        """ Return the variable being transposed
        """
        return self._args[0]

    def __getitem__(self, *args):
        x = self._args[0]
        rank = x.rank
        # Add empty slices to fully index the object
        if len(args) < rank:
            args = args + tuple([Slice(None, None)]*(rank-len(args)))
        return NumpyTranspose(x.__getitem__(*reversed(args)))

    def _set_dtype_precision(self, x):
        self._dtype      = x.dtype
        self._precision  = get_final_precision(x)

    def _set_shape_rank(self, x):
        self._shape      = tuple(reversed(x.shape))
        self._rank       = x.rank

    def _set_order(self, x):
        self._order = 'C' if x.order=='F' else 'F'

    @property
    def is_elemental(self):
        return False

class NumpyConjugate(PythonConjugate):
    """Represents a call to  numpy.conj for code generation.

    > a = 1+2j
    > np.conj(a)
    1-2j
    """
    __slots__ = ('_precision','_rank','_shape','_order')
    name = 'conj'

    def __init__(self, arg):
        super().__init__(arg)
        self._precision = arg.precision
        self._order = arg.order
        self._shape = process_shape(self.internal_var.shape)
        self._rank  = len(self._shape)

    @property
    def is_elemental(self):
        """ Indicates whether the function should be
        called elementwise for an array argument
        """
        return True

#==============================================================================
# TODO split numpy_functions into multiple dictionaries following
# https://docs.scipy.org/doc/numpy-1.15.0/reference/routines.array-creation.html

numpy_linalg_mod = Module('linalg', (),
    [PyccelFunctionDef('norm', NumpyNorm)])

numpy_random_mod = Module('random', (),
    [PyccelFunctionDef('rand'   , NumpyRand),
     PyccelFunctionDef('random' , NumpyRand),
     PyccelFunctionDef('randint', NumpyRandint)])

numpy_constants = {
        'pi': Constant('float', 'pi', value=numpy.pi),
    }

numpy_funcs = {
    # ... array creation routines
    'full'      : PyccelFunctionDef('full'      , NumpyFull),
    'empty'     : PyccelFunctionDef('empty'     , NumpyEmpty),
    'zeros'     : PyccelFunctionDef('zeros'     , NumpyZeros),
    'ones'      : PyccelFunctionDef('ones'      , NumpyOnes),
    'full_like' : PyccelFunctionDef('full_like' , NumpyFullLike),
    'empty_like': PyccelFunctionDef('empty_like', NumpyEmptyLike),
    'zeros_like': PyccelFunctionDef('zeros_like', NumpyZerosLike),
    'ones_like' : PyccelFunctionDef('ones_like' , NumpyOnesLike),
    'array'     : PyccelFunctionDef('array'     , NumpyArray),
    'arange'    : PyccelFunctionDef('arange'    , NumpyArange),
    # ...
    'shape'     : PyccelFunctionDef('shape'     , Shape),
    'norm'      : PyccelFunctionDef('norm'      , NumpyNorm),
    'int'       : PyccelFunctionDef('int'       , NumpyInt),
    'real'      : PyccelFunctionDef('real'      , NumpyReal),
    'imag'      : PyccelFunctionDef('imag'      , NumpyImag),
    'conj'      : PyccelFunctionDef('conj'      , NumpyConjugate),
    'conjugate' : PyccelFunctionDef('conjugate' , NumpyConjugate),
    'float'     : PyccelFunctionDef('float'     , NumpyFloat),
    'double'    : PyccelFunctionDef('double'    , NumpyFloat64),
    'mod'       : PyccelFunctionDef('mod'       , NumpyMod),
    'float32'   : PyccelFunctionDef('float32'   , NumpyFloat32),
    'float64'   : PyccelFunctionDef('float64'   , NumpyFloat64),
    'bool'      : PyccelFunctionDef('bool'      , NumpyBool),
    'int8'      : PyccelFunctionDef('int8'      , NumpyInt8),
    'int16'     : PyccelFunctionDef('int16'     , NumpyInt16),
    'int32'     : PyccelFunctionDef('int32'     , NumpyInt32),
    'int64'     : PyccelFunctionDef('int64'     , NumpyInt64),
    'complex'   : PyccelFunctionDef('complex'   , NumpyComplex),
    'complex128': PyccelFunctionDef('complex128', NumpyComplex128),
    'complex64' : PyccelFunctionDef('complex64' , NumpyComplex64),
    'matmul'    : PyccelFunctionDef('matmul'    , NumpyMatmul),
    'sum'       : PyccelFunctionDef('sum'       , NumpySum),
    'max'       : PyccelFunctionDef('max'       , NumpyAmax),
    'amax'      : PyccelFunctionDef('amax'      , NumpyAmax),
    'min'       : PyccelFunctionDef('min'       , NumpyAmin),
    'amin'      : PyccelFunctionDef('amin'      , NumpyAmin),
    'prod'      : PyccelFunctionDef('prod'      , NumpyProduct),
    'product'   : PyccelFunctionDef('product'   , NumpyProduct),
    'linspace'  : PyccelFunctionDef('linspace'  , NumpyLinspace),
    'where'     : PyccelFunctionDef('where'     , NumpyWhere),
    # ---
    'abs'       : PyccelFunctionDef('abs'       , NumpyAbs),
    'floor'     : PyccelFunctionDef('floor'     , NumpyFloor),
    'absolute'  : PyccelFunctionDef('absolute'  , NumpyAbs),
    'fabs'      : PyccelFunctionDef('fabs'      , NumpyFabs),
    'exp'       : PyccelFunctionDef('exp'       , NumpyExp),
    'log'       : PyccelFunctionDef('log'       , NumpyLog),
    'sqrt'      : PyccelFunctionDef('sqrt'      , NumpySqrt),
    # ---
    'sin'       : PyccelFunctionDef('sin'       , NumpySin),
    'cos'       : PyccelFunctionDef('cos'       , NumpyCos),
    'tan'       : PyccelFunctionDef('tan'       , NumpyTan),
    'arcsin'    : PyccelFunctionDef('arcsin'    , NumpyArcsin),
    'arccos'    : PyccelFunctionDef('arccos'    , NumpyArccos),
    'arctan'    : PyccelFunctionDef('arctan'    , NumpyArctan),
    'arctan2'   : PyccelFunctionDef('arctan2'   , NumpyArctan2),
    # 'hypot'     : PyccelFunctionDef('hypot'     , NumpyHypot),
    'sinh'      : PyccelFunctionDef('sinh'      , NumpySinh),
    'cosh'      : PyccelFunctionDef('cosh'      , NumpyCosh),
    'tanh'      : PyccelFunctionDef('tanh'      , NumpyTanh),
    'arcsinh'   : PyccelFunctionDef('arcsinh'   , NumpyArcsinh),
    'arccosh'   : PyccelFunctionDef('arccosh'   , NumpyArccosh),
    'arctanh'   : PyccelFunctionDef('arctanh'   , NumpyArctanh),
    # 'deg2rad'   : PyccelFunctionDef('deg2rad'   , NumpyDeg2rad),
    # 'rad2deg'   : PyccelFunctionDef('rad2deg'   , NumpyRad2deg),
    'transpose' : PyccelFunctionDef('transpose' , NumpyTranspose),
}

numpy_mod = Module('numpy',
    variables = numpy_constants.values(),
    funcs = numpy_funcs.values(),
    imports = [
        Import('linalg', numpy_linalg_mod),
        Import('random', numpy_random_mod),
        ])

#==============================================================================

numpy_target_swap = {
        numpy_funcs['full_like']  : numpy_funcs['full'],
        numpy_funcs['empty_like'] : numpy_funcs['empty'],
        numpy_funcs['zeros_like'] : numpy_funcs['zeros'],
        numpy_funcs['ones_like']  : numpy_funcs['ones']
    }
