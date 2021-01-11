#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

import numpy

from sympy           import (Tuple, Integer as sp_Integer,
                             Rational as sp_Rational, Expr)

from .core           import (ClassDef, FunctionDef, PyccelInternalFunction,
                             PythonList, Variable, IndexedElement,
                             process_shape, ValuedArgument, Constant)

from .operators      import broadcast

from .builtins       import (PythonInt, PythonBool, PythonFloat, PythonTuple,
                             PythonComplex, PythonReal, PythonImag)

from .datatypes      import (dtype_and_precision_registry as dtype_registry,
                             default_precision, datatype, NativeInteger,
                             NativeReal, NativeComplex, NativeBool, str_dtype)

from .literals       import LiteralInteger, LiteralFloat, LiteralComplex
from .literals       import LiteralTrue, LiteralFalse
from .literals       import Nil
from .basic          import PyccelAstNode


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
    'NumpyZerosLike'
)

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

class NumpyNewArray(PyccelAstNode):

    #--------------------------------------------------------------------------
    @staticmethod
    def _process_order(order):

        if (order is None) or isinstance(order, Nil):
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

    arg : list ,tuple ,Tuple, PythonList

    """

    def __init__(self, arg, dtype=None, order='C'):
        NumpyNewArray.__init__(self)

        if not isinstance(arg, (PythonTuple, PythonList)):
            raise TypeError('Uknown type of  %s.' % type(arg))

        # TODO: treat inhomogenous lists and tuples when they have mixed ordering
        if not arg.is_homogeneous:
            raise TypeError('we only accept a homogeneous list or tuple ')

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

    def _sympystr(self, printer):
        return self.arg

    @property
    def arg(self):
        return self._arg

#==============================================================================
class NumpySum(PyccelInternalFunction):
    """Represents a call to  numpy.sum for code generation.

    arg : list , tuple , PythonTuple, Tuple, PythonList, Variable
    """

    def __init__(self, arg):
        if not isinstance(arg, (list, tuple, PythonTuple, Tuple, PythonList,
                            Variable, Expr)):
            raise TypeError('Uknown type of  %s.' % type(arg))
        PyccelInternalFunction.__init__(self, arg)
        self._dtype = arg.dtype
        self._rank  = 0
        self._shape = ()
        self._precision = default_precision[str_dtype(self._dtype)]

    @property
    def arg(self):
        return self._args[0]

#==============================================================================
class NumpyProduct(PyccelInternalFunction):
    """Represents a call to  numpy.prod for code generation.

    arg : list , tuple , PythonTuple, Tuple, PythonList, Variable
    """

    def __init__(self, arg):
        if not isinstance(arg, (list, tuple, PythonTuple, Tuple, PythonList,
                                Variable, Expr)):
            raise TypeError('Uknown type of  %s.' % type(arg))
        PyccelInternalFunction.__init__(self, arg)
        self._dtype = arg.dtype
        self._rank  = 0
        self._shape = ()
        self._precision = default_precision[str_dtype(self._dtype)]

    @property
    def arg(self):
        return self._args[0]


#==============================================================================
class NumpyMatmul(PyccelInternalFunction):
    """Represents a call to numpy.matmul for code generation.
    arg : list , tuple , PythonTuple, Tuple, PythonList, Variable
    """

    def __init__(self, a ,b):
        if not isinstance(a, (list, tuple, PythonTuple, Tuple, PythonList,
                                Variable, Expr)):
            raise TypeError('Unknown type of  %s.' % type(a))
        if not isinstance(b, (list, tuple, PythonTuple, Tuple, PythonList,
                                Variable, Expr)):
            raise TypeError('Unknown type of  %s.' % type(a))
        PyccelInternalFunction.__init__(self, a, b)

        args      = (a, b)
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

        if a.rank == 1 or b.rank == 1:
            self._rank = 1
        else:
            self._rank = 2

        if not (a.shape is None or b.shape is None):

            m = 1 if a.rank < 2 else a.shape[0]
            n = 1 if b.rank < 2 else b.shape[1]
            self._shape = (m, n)

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

def Shape(arg):
    if isinstance(arg.shape, PythonTuple):
        return arg.shape
    else:
        return PythonTuple(*arg.shape)

#==============================================================================
class NumpyReal(PythonReal):
    """Represents a call to  numpy.real for code generation.

    > a = 1+2j
    > np.real(a)
    1.0
    """

#==============================================================================
class NumpyImag(PythonImag):
    """Represents a call to  numpy.real for code generation.

    > a = 1+2j
    > np.imag(a)
    2.0
    """

#==============================================================================
class NumpyLinspace(NumpyNewArray):

    """
    Represents numpy.linspace.

    """
    _dtype     = NativeReal()
    _precision = default_precision['real']
    _rank      = 1
    _order     = 'F'

    def __init__(self, start, stop, size):


        _valid_args = (Variable, IndexedElement, LiteralFloat,
                       sp_Integer, sp_Rational)

        for arg in (start, stop, size):
            if not isinstance(arg, _valid_args):
                raise TypeError('Expecting valid args')

        self._index = Variable('int', 'linspace_index')
        self._start = start
        self._stop  = stop
        self._size  = size
        self._shape = (self.size,)
        NumpyNewArray.__init__(self)

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

    def _sympystr(self, printer):
        sstr = printer.doprint
        code = 'linspace({}, {}, {})',format(sstr(self.start),
                                             sstr(self.stop),
                                             sstr(self.size))
        return code

#==============================================================================
class NumpyWhere(PyccelInternalFunction):
    """ Represents a call to  numpy.where """

    def __init__(self, mask):
        PyccelInternalFunction.__init__(self, mask)


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
    _dtype = NativeReal()
    _precision = default_precision['real']

    def __init__(self, *args):
        PyccelInternalFunction.__init__(self)
        self._shape = args
        self._rank  = len(self.shape)

    @property
    def order(self):
        return 'C'

#==============================================================================
class NumpyRandint(PyccelInternalFunction):

    """
      Represents a call to  numpy.random.random or numpy.random.rand for code generation.

    """
    _dtype = NativeInteger()
    _precision = default_precision['integer']
    _order = 'C'

    def __init__(self, low, high = None, size = None):
        PyccelInternalFunction.__init__(self)
        if size is None:
            size = ()
        if not hasattr(size,'__iter__'):
            size = (size,)

        self._shape   = size
        self._rank    = len(self.shape)
        self._rand    = NumpyRand(*size)
        self._low     = low
        self._high    = high

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
class NumpyFull(PyccelInternalFunction, NumpyNewArray):
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

    def __init__(self, shape, fill_value, dtype=None, order='C'):

        # Convert shape to PythonTuple
        shape = process_shape(shape)

        # If there is no dtype, extract it from fill_value
        # TODO: must get dtype from an annotated node
        if (dtype is None) or isinstance(dtype, Nil):
            dtype = fill_value.dtype

        # Verify dtype and get precision
        dtype, precision = process_dtype(dtype)

        # Verify array ordering
        order = NumpyNewArray._process_order(order)

        # Cast fill_value to correct type
        # TODO [YG, 09.11.2020]: treat difficult case of LiteralComplex
        from pyccel.ast.datatypes import str_dtype
        stype = str_dtype(dtype)
        if fill_value and stype != 'complex':
            from pyccel.codegen.printing.fcode import python_builtin_datatypes
            cast_func  = python_builtin_datatypes[stype]
            fill_value = cast_func(fill_value)
        self._shape = shape
        self._rank  = len(self._shape)
        self._dtype = dtype
        self._order = order
        self._precision = precision

        PyccelInternalFunction.__init__(self, fill_value)
        NumpyNewArray.__init__(self)

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
        if (dtype is None) or isinstance(dtype, Nil):
            raise TypeError("Data type must be provided")

        NumpyFull.__init__(self, shape, None, dtype, order)

#==============================================================================
class NumpyEmpty(NumpyAutoFill):
    """ Represents a call to numpy.empty for code generation.
    """
    @property
    def fill_value(self):
        return None


#==============================================================================
class NumpyZeros(NumpyAutoFill):
    """ Represents a call to numpy.zeros for code generation.
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
class NumpyOnes(NumpyAutoFill):
    """ Represents a call to numpy.ones for code generation.
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
class NumpyFullLike:
    """ Represents a call to numpy.full_like for code generation.
    """
    def __new__(cls, a, fill_value, dtype=None, order='K', subok=True):

        # NOTE: we ignore 'subok' argument
        dtype = a.dtype if (dtype is None) or isinstance(dtype, Nil) else dtype
        order = a.order if str(order).strip('\'"') in ('K', 'A') else order

        return NumpyFull(Shape(a), fill_value, dtype, order)

#=======================================================================================
class NumpyEmptyLike:
    """ Represents a call to numpy.empty_like for code generation.
    """
    def __new__(cls, a, dtype=None, order='K', subok=True):

        # NOTE: we ignore 'subok' argument
        dtype = a.dtype if (dtype is None) or isinstance(dtype, Nil) else dtype
        order = a.order if str(order).strip('\'"') in ('K', 'A') else order

        return NumpyEmpty(Shape(a), dtype, order)

#=======================================================================================
class NumpyOnesLike:
    """ Represents a call to numpy.ones_like for code generation.
    """
    def __new__(cls, a, dtype=None, order='K', subok=True):

        # NOTE: we ignore 'subok' argument
        dtype = a.dtype if (dtype is None) or isinstance(dtype, Nil) else dtype
        order = a.order if str(order).strip('\'"') in ('K', 'A') else order

        return NumpyOnes(Shape(a), dtype, order)

#=======================================================================================
class NumpyZerosLike:
    """ Represents a call to numpy.zeros_like for code generation.
    """
    def __new__(cls, a, dtype=None, order='K', subok=True):

        # NOTE: we ignore 'subok' argument
        dtype = a.dtype if (dtype is None) or isinstance(dtype, Nil) else dtype
        order = a.order if str(order).strip('\'"') in ('K', 'A') else order

        return NumpyZeros(Shape(a), dtype, order)

#=======================================================================================

class NumpyNorm(PyccelInternalFunction):
    """ Represents call to numpy.norm"""
    _dtype = NativeReal()

    def __init__(self, arg, dim=None):
        if isinstance(dim, ValuedArgument):
            dim = dim.value
        if self.dim is not None:
            sh = list(sh)
            del sh[self.dim]
            self._shape = tuple(sh)
        else:
            self._shape = ()
        self._rank = len(self._shape)
        PyccelInternalFunction.__init__(self, arg, dim)

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

#------------------------------------------------------------------------------
class NumpyUfuncUnary(NumpyUfuncBase):
    """Numpy's universal function with one argument.
    """
    def __init__(self, x):
        self._set_dtype_precision(x)
        self._set_shape_rank(x)
        self._order      = x.order
        super().__init__(x)

    def _set_shape_rank(self, x):
        self._shape      = x.shape
        self._rank       = x.rank

    def _set_dtype_precision(self, x):
        self._dtype      = x.dtype if x.dtype is NativeComplex() else NativeReal()
        self._precision  = default_precision[str_dtype(self._dtype)]

#------------------------------------------------------------------------------
class NumpyUfuncBinary(NumpyUfuncBase):
    """Numpy's universal function with two arguments.
    """
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
        self._dtype     = NativeReal()
        self._precision = default_precision['real']

    def _set_order(self, x1, x2):
        if x1.order == x2.order:
            self._order = x1.order
        else:
            self._order = 'C'

#------------------------------------------------------------------------------
# Math operations
#------------------------------------------------------------------------------
#class NumpyAbsolute(NumpyUfuncUnary): pass
class NumpyFabs    (NumpyUfuncUnary): pass
class NumpyExp     (NumpyUfuncUnary): pass
class NumpyLog     (NumpyUfuncUnary): pass
class NumpySqrt    (NumpyUfuncUnary): pass

#------------------------------------------------------------------------------
# Trigonometric functions
#------------------------------------------------------------------------------
class NumpySin    (NumpyUfuncUnary) : pass
class NumpyCos    (NumpyUfuncUnary) : pass
class NumpyTan    (NumpyUfuncUnary) : pass
class NumpyArcsin (NumpyUfuncUnary) : pass
class NumpyArccos (NumpyUfuncUnary) : pass
class NumpyArctan (NumpyUfuncUnary) : pass
class NumpyArctan2(NumpyUfuncBinary): pass
class NumpyHypot  (NumpyUfuncBinary): pass
class NumpySinh   (NumpyUfuncUnary) : pass
class NumpyCosh   (NumpyUfuncUnary) : pass
class NumpyTanh   (NumpyUfuncUnary) : pass
class NumpyArcsinh(NumpyUfuncUnary) : pass
class NumpyArccosh(NumpyUfuncUnary) : pass
class NumpyArctanh(NumpyUfuncUnary) : pass
#class NumpyDeg2rad(NumpyUfuncUnary) : pass
#class NumpyRad2deg(NumpyUfuncUnary) : pass

#=======================================================================================

class NumpyAbs(NumpyUfuncUnary):
    def _set_dtype_precision(self, x):
        self._dtype     = NativeInteger() if x.dtype is NativeInteger() else NativeReal()
        self._precision = default_precision[str_dtype(self._dtype)]

class NumpyFloor(NumpyUfuncUnary):
    def _set_dtype_precision(self, x):
        self._dtype     = NativeReal()
        self._precision = default_precision[str_dtype(self._dtype)]

class NumpyMod(NumpyUfuncBinary):

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
    def _set_shape_rank(self, x):
        self._shape     = ()
        self._rank      = 0

    def _set_dtype_precision(self, x):
        self._dtype     = x.dtype
        self._precision = x.precision

class NumpyMax(NumpyUfuncUnary):
    def _set_shape_rank(self, x):
        self._shape     = ()
        self._rank      = 0

    def _set_dtype_precision(self, x):
        self._dtype     = x.dtype
        self._precision = x.precision


#=======================================================================================
class NumpyComplex(PythonComplex):
    """ Represents a call to numpy.complex() function.
    """

class NumpyComplex64(NumpyComplex):
    _precision = dtype_registry['complex64'][1]

class NumpyComplex128(NumpyComplex):
    _precision = dtype_registry['complex128'][1]

#=======================================================================================
class NumpyFloat(PythonFloat):
    """ Represents a call to numpy.float() function.
    """

class NumpyFloat32(NumpyFloat):
    """ Represents a call to numpy.float32() function.
    """
    _precision = dtype_registry['float32'][1]

class NumpyFloat64(NumpyFloat):
    """ Represents a call to numpy.float64() function.
    """
    _precision = dtype_registry['float64'][1]

#=======================================================================================
# TODO [YG, 13.03.2020]: handle case where base != 10
class NumpyInt(PythonInt):
    """ Represents a call to numpy.int() function.
    """
    def __new__(cls, arg=None, base=10):
        return PythonInt.__new__(cls, arg)

class NumpyInt32(NumpyInt):
    """ Represents a call to numpy.int32() function.
    """
    _precision = dtype_registry['int32'][1]

class NumpyInt64(NumpyInt):
    """ Represents a call to numpy.int64() function.
    """
    _precision = dtype_registry['int64'][1]



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
