#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

import numpy

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
from .literals       import Nil, convert_to_literal
from .mathext        import MathCeil
from .operators      import broadcast, PyccelMinus, PyccelDiv
from .variable       import (Variable, IndexedElement, Constant)


__all__ = (
    'NumpyArrayClass',
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
    'NumpyInt',
    'NumpyInt32',
    'NumpyInt64',
    'NumpyLinspace',
    'NumpyMatmul',
    'NumpyMax',
    'NumpyMin',
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
class NumpyComplex(PythonComplex):
    """ Represents a call to numpy.complex() function.
    """
    __slots__ = ()

class NumpyComplex64(NumpyComplex):
    """ Represents a call to numpy.complex64() function.
    """
    __slots__ = ()
    def _set_dtype(self):
        self._dtype = NativeComplex()
        self._precision = dtype_registry['complex64'][1]

class NumpyComplex128(NumpyComplex):
    """ Represents a call to numpy.complex128() function.
    """
    __slots__ = ()
    def _set_dtype(self):
        self._dtype = NativeComplex()
        self._precision = dtype_registry['complex128'][1]

#=======================================================================================
class NumpyFloat(PythonFloat):
    """ Represents a call to numpy.float() function.
    """
    __slots__ = ()

class NumpyFloat32(NumpyFloat):
    """ Represents a call to numpy.float32() function.
    """
    __slots__ = ()
    def _set_dtype(self):
        self._dtype = NativeReal()
        self._precision = dtype_registry['float32'][1]

class NumpyFloat64(NumpyFloat):
    """ Represents a call to numpy.float64() function.
    """
    __slots__ = ()
    def _set_dtype(self):
        self._dtype = NativeReal()
        self._precision = dtype_registry['float64'][1]

#=======================================================================================
# TODO [YG, 13.03.2020]: handle case where base != 10
class NumpyInt(PythonInt):
    """ Represents a call to numpy.int() function.
    """
    __slots__ = ()
    def __new__(cls, arg=None, base=10):
        return super().__new__(cls, arg)

class NumpyInt32(NumpyInt):
    """ Represents a call to numpy.int32() function.
    """
    __slots__ = ()
    def _set_dtype(self):
        self._dtype = NativeInteger()
        self._precision = dtype_registry['int32'][1]

class NumpyInt64(NumpyInt):
    """ Represents a call to numpy.int64() function.
    """
    __slots__ = ()
    def _set_dtype(self):
        self._dtype = NativeInteger()
        self._precision = dtype_registry['int64'][1]

#==============================================================================
class NumpyReal(PythonReal):
    """Represents a call to  numpy.real for code generation.

    > a = 1+2j
    > np.real(a)
    1.0
    """
    __slots__ = ()
    def __init__(self, arg):
        super().__init__(arg)

    def _set_shape(self):
        self._shape = process_shape(self.internal_var.shape)

    def _set_order(self):
        self._order = self.internal_var.order

#==============================================================================
DtypePrecisionToCastFunction = {
    'Int' : {
        4 : NumpyInt32,
        8 : NumpyInt64},
    'Real' : {
        4 : NumpyFloat32,
        8 : NumpyFloat64},
    'Complex' : {
        4 : NumpyComplex64,
        8 : PythonComplex,
        16 : NumpyComplex128,},
    'Bool':  {
        4 : PythonBool}
}

#==============================================================================
numpy_constants = {
    'pi': Constant('real', 'pi', value=numpy.pi),
}

def process_dtype(dtype):
    if dtype  in (PythonInt, PythonFloat, PythonComplex, PythonBool):
        # remove python prefix from dtype.name len("python") = 6
        dtype = dtype.__name__.lower()[6:]
    elif dtype  in (NumpyInt, NumpyInt32, NumpyInt64, NumpyComplex, NumpyFloat,
				  NumpyComplex128, NumpyComplex64, NumpyFloat64, NumpyFloat32):
        # remove numpy prefix from dtype.name len("numpy") = 5
        dtype = dtype.__name__.lower()[5:]
    else:
        dtype            = str(dtype).replace('\'', '').lower()
    dtype, precision = dtype_registry[dtype]
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
    __slots__ = ('_arg',)
    _attribute_nodes = ('_arg',)

    def __init__(self, arg, dtype=None, order='C'):

        if not isinstance(arg, (PythonTuple, PythonList, Variable)):
            raise TypeError('Unknown type of  %s.' % type(arg))

        # TODO: treat inhomogenous lists and tuples when they have mixed ordering
        if isinstance(arg, (PythonTuple, PythonList)) and not arg.is_homogeneous or \
            isinstance(arg, Variable) and not arg.is_ndarray and not arg.is_stack_array:
            raise TypeError('we only accept homogeneous arguments')

        # ...
        self._arg   = arg
        self._dtype = dtype
        self._order = order
        super().__init__()

    def _set_dtype(self):
        # Verify dtype and get precision
        if self._dtype is None:
            self._dtype = self.arg.dtype
        dtype, prec = process_dtype(self._dtype)
        self._dtype = dtype
        self._precision = prec

    def _set_shape(self):
        self._shape = process_shape(self.arg.shape)

    def _set_order(self):
        # ... Determine ordering
        order = self._order
        if isinstance(order, ValuedArgument):
            order = order.value
        order = str(order).strip("\'")

        if order not in ('K', 'A', 'C', 'F'):
            raise ValueError("Cannot recognize '{:s}' order".format(order))

        # TODO [YG, 18.02.2020]: set correct order based on input array
        if order in ('K', 'A'):
            order = 'C'
        self._order = order

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
    __slots__ = ('_start','_step','_stop')
    _attribute_nodes = ('_start','_step','_stop')

    def __init__(self, start, stop = None, step = None, dtype = None):

        if stop is None:
            self._start = LiteralInteger(0)
            self._stop = start
        else:
            self._start = start
            self._stop = stop
        self._step = step if step is not None else LiteralInteger(1)

        self._dtype = dtype

        super().__init__()

    def _set_dtype(self):
        if self._dtype is None:
            self._dtype = max([i.dtype for i in self.arg], key = NativeNumeric.index)
            self._precision = max([i.precision for i in self.arg])
        else:
            self._dtype, self._precision = process_dtype(self._dtype)

    def _set_shape(self):
        self._shape = (MathCeil(PyccelDiv(PyccelMinus(self._stop, self._start), self._step)))
        self._shape = process_shape(self._shape)

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
    __slots__ = ()

    def __init__(self, arg):
        if not isinstance(arg, PyccelAstNode):
            raise TypeError('Unknown type of  %s.' % type(arg))
        super().__init__(arg)

    def _set_dtype(self):
        self._dtype = self.arg.dtype

    def _set_shape(self):
        self._shape = ()

    @property
    def arg(self):
        return self._args[0]

#==============================================================================
class NumpyProduct(PyccelInternalFunction):
    """Represents a call to  numpy.prod for code generation.

    arg : list , tuple , PythonTuple, PythonList, Variable
    """
    __slots__ = ()

    def __init__(self, arg):
        if not isinstance(arg, PyccelAstNode):
            raise TypeError('Unknown type of  %s.' % type(arg))
        super().__init__(arg)

    def _set_dtype(self):
        self._dtype = self.arg.dtype

    def _set_shape(self):
        self._shape = ()

    @property
    def arg(self):
        return self._args[0]


#==============================================================================
class NumpyMatmul(PyccelInternalFunction):
    """Represents a call to numpy.matmul for code generation.
    arg : list , tuple , PythonTuple, PythonList, Variable
    """
    __slots__ = ()

    def __init__(self, a ,b):
        if not isinstance(a, PyccelAstNode):
            raise TypeError('Unknown type of  %s.' % type(a))
        if not isinstance(b, PyccelAstNode):
            raise TypeError('Unknown type of  %s.' % type(a))
        super().__init__(a, b)

    def _set_dtype(self):
        args      = self.args

        integers  = [e for e in args if e.dtype is NativeInteger() or a.dtype is NativeBool()]
        reals     = [e for e in args if e.dtype is NativeReal()]
        complexs  = [e for e in args if e.dtype is NativeComplex()]

        if complexs:
            self._dtype     = NativeComplex()
            self._precision = max(e.precision for e in complexs)
        if reals:
            self._dtype     = NativeReal()
            self._precision = max(e.precision for e in reals)
        elif integers:
            self._dtype     = NativeInteger()
            self._precision = max(e.precision for e in integers)
        else:
            raise TypeError('cannot determine the type of {}'.format(self))

    def _set_shape(self):
        if not (self.a.shape is None or self.b.shape is None):

            m = 1 if a.rank < 2 else a.shape[0]
            n = 1 if b.rank < 2 else b.shape[1]
            self._shape = (m, n)

    def _set_rank(self):
        if self.a.rank == 1 or self.b.rank == 1:
            self._rank = 1
        else:
            self._rank = 2

    def _set_order(self):
        if self.a.order == self.b.order:
            self._order = self.a.order
        else:
            self._order = 'C'

    @property
    def a(self):
        return self._args[0]

    @property
    def b(self):
        return self._args[1]


#==============================================================================

def Shape(arg):
    if isinstance(arg.shape, PythonTuple):
        return arg.shape
    else:
        return PythonTuple(*arg.shape)

#==============================================================================
class NumpyImag(PythonImag):
    """Represents a call to  numpy.real for code generation.

    > a = 1+2j
    > np.imag(a)
    2.0
    """
    __slots__ = ()

#==============================================================================
class NumpyLinspace(NumpyNewArray):

    """
    Represents numpy.linspace.

    """
    __slots__ = ('_index','_start','_stop','_size')

    def __init__(self, start, stop, size):


        _valid_args = (Variable, IndexedElement, LiteralFloat,
                       LiteralInteger)

        for arg in (start, stop, size):
            if not isinstance(arg, _valid_args):
                raise TypeError('Expecting valid args')

        self._index = Variable('int', 'linspace_index')
        self._start = start
        self._stop  = stop
        self._size  = size
        super().__init__()

    def _set_dtype(self):
        _dtype     = NativeReal()

    def _set_shape(self):
        self._shape = (self.size,)

    @property
    def start(self):
        return self._start

    @property
    def stop(self):
        return self._stop

    @property
    def size(self):
        return self._size

    @property
    def index(self):
        return self._index

    @property
    def step(self):
        return (self.stop - self.start) / (self.size - 1)

    def __str__(self):
        code = 'linspace({}, {}, {})'.format(str(self.start),
                                             str(self.stop),
                                             str(self.size))
        return code

#==============================================================================
class NumpyWhere(PyccelInternalFunction):
    """ Represents a call to  numpy.where """
    __slots__ = ()

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
    __slots__ = ()

    def __init__(self, *args):
        super().__init__(*args)
        self._shape = args

    def _set_dtype(self):
        self._dtype = NativeReal()

    def _set_shape(self):
        pass

    @property
    def order(self):
        return 'C'

#==============================================================================
class NumpyRandint(PyccelInternalFunction):

    """
      Represents a call to  numpy.random.random or numpy.random.rand for code generation.

    """
    __slots__ = ('_rand','_low','_high')
    _attribute_nodes = ('_low', '_high')

    def __init__(self, low, high = None, size = None):
        if size is None:
            self._shape = ()
        elif not hasattr(size,'__iter__'):
            self._shape = (size,)
        else:
            self._shape = size

        self._rand    = NumpyRand(*size)
        self._low     = low
        self._high    = high
        super().__init__()

    def _set_dtype(self):
        self._dtype = NativeInteger()

    def _set_shape(self):
        pass

    def _set_order(self):
        self._order = 'C'

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
    __slots__ = ()

    def __init__(self, shape, fill_value, dtype=None, order='C'):
        self._dtype = dtype
        self._shape = shape
        self._order = order

        # If there is no dtype, extract it from fill_value
        # TODO: must get dtype from an annotated node
        if not self._dtype:
            self._dtype = self.fill_value.dtype
        # Verify dtype and get precision
        dtype, precision = process_dtype(dtype)
        self._dtype = dtype
        self._precision = precision

        # Cast fill_value to correct type
        if fill_value and not isinstance(fill_value, Nil) and dtype != fill_value.dtype:
            cast_func = DtypePrecisionToCastFunction[dtype.name][precision]
            fill_value = cast_func(fill_value)

        super().__init__(fill_value)

    def _set_dtype(self):
        pass

        # Verify dtype and get precision
        dtype, precision = process_dtype(self._dtype)
        self._dtype = dtype
        self._precision = precision

    def _set_shape(self):
        # Convert shape to PythonTuple
        self._shape = process_shape(self._shape)

    def _set_order(self):
        # Verify array ordering
        self._order = NumpyNewArray._process_order(self._order)

    #--------------------------------------------------------------------------
    @property
    def fill_value(self):
        return self._args[0]

#==============================================================================
class NumpyAutoFill(NumpyFull):
    """ Abstract class for all classes which inherit from NumpyFull but
        the fill_value is implicitly specified
    """
    def __init__(self, shape, dtype='float', order='C'):
        if not dtype:
            raise TypeError("Data type must be provided")

        super().__init__(shape, Nil(), dtype, order)

#==============================================================================
class NumpyEmpty(NumpyAutoFill):
    """ Represents a call to numpy.empty for code generation.
    """
    __slots__ = ()
    @property
    def fill_value(self):
        return None


#==============================================================================
class NumpyZeros(NumpyAutoFill):
    """ Represents a call to numpy.zeros for code generation.
    """
    __slots__ = ()
    @property
    def fill_value(self):
        return convert_to_literal(0, self.dtype, self.precision)

#==============================================================================
class NumpyOnes(NumpyAutoFill):
    """ Represents a call to numpy.ones for code generation.
    """
    __slots__ = ()
    @property
    def fill_value(self):
        return convert_to_literal(1, self.dtype, self.precision)

#=======================================================================================
class NumpyFullLike:
    """ Represents a call to numpy.full_like for code generation.
    """
    __slots__ = ()
    def __new__(cls, a, fill_value, dtype=None, order='K', subok=True, shape=None):

        # NOTE: we ignore 'subok' argument
        dtype = dtype or a.dtype
        order = a.order if str(order).strip('\'"') in ('K', 'A') else order
        shape = Shape(a) if shape is None else shape
        return NumpyFull(shape, fill_value, dtype, order)

#=======================================================================================
class NumpyEmptyLike:
    """ Represents a call to numpy.empty_like for code generation.
    """
    __slots__ = ()
    def __new__(cls, a, dtype=None, order='K', subok=True, shape=None):

        # NOTE: we ignore 'subok' argument
        dtype = dtype or a.dtype
        order = a.order if str(order).strip('\'"') in ('K', 'A') else order
        shape = Shape(a) if shape is None else shape

        return NumpyEmpty(shape, dtype, order)

#=======================================================================================
class NumpyOnesLike:
    """ Represents a call to numpy.ones_like for code generation.
    """
    __slots__ = ()
    def __new__(cls, a, dtype=None, order='K', subok=True, shape=None):

        # NOTE: we ignore 'subok' argument
        dtype = dtype or a.dtype
        order = a.order if str(order).strip('\'"') in ('K', 'A') else order
        shape = Shape(a) if shape is None else shape

        return NumpyOnes(shape, dtype, order)

#=======================================================================================
class NumpyZerosLike:
    """ Represents a call to numpy.zeros_like for code generation.
    """
    __slots__ = ()
    def __new__(cls, a, dtype=None, order='K', subok=True, shape=None):

        # NOTE: we ignore 'subok' argument
        dtype = dtype or a.dtype
        order = a.order if str(order).strip('\'"') in ('K', 'A') else order
        shape = Shape(a) if shape is None else shape

        return NumpyZeros(shape, dtype, order)

#=======================================================================================

class NumpyNorm(PyccelInternalFunction):
    """ Represents call to numpy.norm"""
    __slots__ = ()

    def __init__(self, arg, dim=None):
        if isinstance(dim, ValuedArgument):
            dim = dim.value
        if self.dim is not None:
            sh = list(sh)
            del sh[self.dim]
            self._shape = tuple(sh)
        else:
            self._shape = ()
        super().__init__(arg, dim)

    def _set_dtype(self):
        self._dtype = NativeReal()

    def _set_shape(self):
        pass

    @property
    def arg(self):
        return self._args[0]

    @property
    def dim(self):
        return self._args[1]

#====================================================


#==============================================================================
# Numpy universal functions
# https://numpy.org/doc/stable/reference/ufuncs.html#available-ufuncs
#==============================================================================
class NumpyUfuncBase(PyccelInternalFunction):
    """Base class for Numpy's universal functions."""
    __slots__ = ()
    @property
    def is_elemental(self):
        return True

#------------------------------------------------------------------------------
class NumpyUfuncUnary(NumpyUfuncBase):
    """Numpy's universal function with one argument.
    """
    __slots__ = ()
    def __init__(self, x):
        super().__init__(x)

    def _set_dtype_precision(self):
        x = self.args[0]
        self._dtype = x.dtype if x.dtype is NativeComplex() else NativeReal()

    def _set_shape(self, x):
        self._shape = self.args[0].shape

    def _set_order(self):
        self._order = self.args[0].order

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

    def _set_dtype_precision(self):
        self._dtype     = NativeReal()

    def _set_shape(self):
        self._shape = self.args[0].shape  # TODO ^^

    def _set_order(self):
        x1 = self.args[0]
        x2 = self.args[1]
        if x1.order == x2.order:
            self._order = x1.order
        else:
            self._order = 'C'

#------------------------------------------------------------------------------
# Math operations
#------------------------------------------------------------------------------
#class NumpyAbsolute(NumpyUfuncUnary): __slots__ = ()
class NumpyFabs    (NumpyUfuncUnary): __slots__ = ()
class NumpyExp     (NumpyUfuncUnary): __slots__ = ()
class NumpyLog     (NumpyUfuncUnary): __slots__ = ()
class NumpySqrt    (NumpyUfuncUnary): __slots__ = ()

#------------------------------------------------------------------------------
# Trigonometric functions
#------------------------------------------------------------------------------
class NumpySin    (NumpyUfuncUnary) : __slots__ = ()
class NumpyCos    (NumpyUfuncUnary) : __slots__ = ()
class NumpyTan    (NumpyUfuncUnary) : __slots__ = ()
class NumpyArcsin (NumpyUfuncUnary) : __slots__ = ()
class NumpyArccos (NumpyUfuncUnary) : __slots__ = ()
class NumpyArctan (NumpyUfuncUnary) : __slots__ = ()
class NumpyArctan2(NumpyUfuncBinary): __slots__ = ()
class NumpyHypot  (NumpyUfuncBinary): __slots__ = ()
class NumpySinh   (NumpyUfuncUnary) : __slots__ = ()
class NumpyCosh   (NumpyUfuncUnary) : __slots__ = ()
class NumpyTanh   (NumpyUfuncUnary) : __slots__ = ()
class NumpyArcsinh(NumpyUfuncUnary) : __slots__ = ()
class NumpyArccosh(NumpyUfuncUnary) : __slots__ = ()
class NumpyArctanh(NumpyUfuncUnary) : __slots__ = ()
#class NumpyDeg2rad(NumpyUfuncUnary) : __slots__ = ()
#class NumpyRad2deg(NumpyUfuncUnary) : __slots__ = ()

#=======================================================================================

class NumpyAbs(NumpyUfuncUnary):
    __slots__ = ()
    def _set_dtype(self):
        x = self.args[0]
        self._dtype     = NativeInteger() if x.dtype is NativeInteger() else NativeReal()

class NumpyFloor(NumpyUfuncUnary):
    __slots__ = ()
    def _set_dtype(self):
        self._dtype     = NativeReal()

class NumpyMod(NumpyUfuncBinary):
    __slots__ = ()

    def _set_shape(self):
        shapes = [a.shape for a in self.args]

        if len(args) == 1:
            shape = self.args[0].shape
        else:
            shape = broadcast(self.args[0].shape, self.args[1].shape)

            for a in self.args[2:]:
                shape = broadcast(shape, a.shape)

        self._shape = shape

    def _set_dtype(self, x1, x2):
        args      = (x1, x2)
        integers  = [a for a in args if a.dtype is NativeInteger() or a.dtype is NativeBool()]
        reals     = [a for a in args if a.dtype is NativeReal()]
        others    = [a for a in args if a not in integers+reals]

        if others:
            raise TypeError('{} not supported'.format(others[0].dtype))

        if reals:
            self._dtype     = NativeReal()
            self._precision = max(a.precision for a in reals)
        elif integers:
            self._dtype     = NativeInteger()
            self._precision = max(a.precision for a in integers)
        else:
            raise TypeError('cannot determine the type of {}'.format(self))

class NumpyMin(NumpyUfuncUnary):
    __slots__ = ()
    def _set_shape(self):
        self._shape     = ()

    def _set_dtype(self):
        x = self.args[0]
        self._dtype     = x.dtype
        self._precision = x.precision

    @property
    def is_elemental(self):
        return False

class NumpyMax(NumpyUfuncUnary):
    __slots__ = ()
    def _set_shape(self):
        self._shape     = ()

    def _set_dtype(self):
        x = self.args[0]
        self._dtype     = x.dtype
        self._precision = x.precision

    @property
    def is_elemental(self):
        return False


#=======================================================================================


NumpyArrayClass = ClassDef('numpy.ndarray',
        methods=[
            FunctionDef('shape',[],[],body=[],
                decorators={'property':'property', 'numpy_wrapper':Shape}),
            FunctionDef('sum',[],[],body=[],
                decorators={'numpy_wrapper':NumpySum}),
            FunctionDef('min',[],[],body=[],
                decorators={'numpy_wrapper':NumpyMin}),
            FunctionDef('max',[],[],body=[],
                decorators={'numpy_wrapper':NumpyMax}),
            FunctionDef('imag',[],[],body=[],
                decorators={'property':'property', 'numpy_wrapper':NumpyImag}),
            FunctionDef('real',[],[],body=[],
                decorators={'property':'property', 'numpy_wrapper':NumpyReal})])

#==============================================================================
# TODO split numpy_functions into multiple dictionaries following
# https://docs.scipy.org/doc/numpy-1.15.0/reference/routines.array-creation.html
numpy_functions = {
    # ... array creation routines
    'full'      : NumpyFull,
    'empty'     : NumpyEmpty,
    'zeros'     : NumpyZeros,
    'ones'      : NumpyOnes,
    'full_like' : NumpyFullLike,
    'empty_like': NumpyEmptyLike,
    'zeros_like': NumpyZerosLike,
    'ones_like' : NumpyOnesLike,
    'array'     : NumpyArray,
    'arange'    : NumpyArange,
    # ...
    'shape'     : Shape,
    'norm'      : NumpyNorm,
    'int'       : NumpyInt,
    'real'      : NumpyReal,
    'imag'      : NumpyImag,
    'float'     : NumpyFloat,
    'double'    : NumpyFloat64,
    'mod'       : NumpyMod,
    'float32'   : NumpyFloat32,
    'float64'   : NumpyFloat64,
    'int32'     : NumpyInt32,
    'int64'     : NumpyInt64,
    'complex'   : NumpyComplex,
    'complex128': NumpyComplex128,
    'complex64' : NumpyComplex64,
    'matmul'    : NumpyMatmul,
    'sum'       : NumpySum,
    'max'      : NumpyMax,
    'min'      : NumpyMin,
    'prod'      : NumpyProduct,
    'product'   : NumpyProduct,
    'linspace'  : NumpyLinspace,
    'where'     : NumpyWhere,
    # ---
    'abs'       : NumpyAbs,
    'floor'     : NumpyFloor,
    'absolute'  : NumpyAbs,
    'fabs'      : NumpyFabs,
    'exp'       : NumpyExp,
    'log'       : NumpyLog,
    'sqrt'      : NumpySqrt,
    # ---
    'sin'       : NumpySin,
    'cos'       : NumpyCos,
    'tan'       : NumpyTan,
    'arcsin'    : NumpyArcsin,
    'arccos'    : NumpyArccos,
    'arctan'    : NumpyArctan,
    'arctan2'   : NumpyArctan2,
    # 'hypot'     : NumpyHypot,
    'sinh'      : NumpySinh,
    'cosh'      : NumpyCosh,
    'tanh'      : NumpyTanh,
    'arcsinh'   : NumpyArcsinh,
    'arccosh'   : NumpyArccosh,
    'arctanh'   : NumpyArctanh,
    # 'deg2rad'   : NumpyDeg2rad,
    # 'rad2deg'   : NumpyRad2deg,
}

numpy_linalg_functions = {
    'norm'      : NumpyNorm,
}

numpy_random_functions = {
    'rand'      : NumpyRand,
    'random'    : NumpyRand,
    'randint'   : NumpyRandint,
}
