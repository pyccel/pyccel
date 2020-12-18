#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

import numpy

from sympy.core.function import Application

from sympy           import (Basic, Function, Tuple, Integer as sp_Integer,
                             Rational as sp_Rational, Expr)

from .core           import (ClassDef, FunctionDef,
                             PythonList, Variable, IndexedElement,
                             Nil, process_shape, ValuedArgument, Constant)

from .operators      import (PyccelPow, PyccelAssociativeParenthesis, broadcast)

from .builtins       import (PythonInt, PythonBool, PythonFloat, PythonTuple,
                             PythonComplex, PythonReal, PythonImag)

from .datatypes      import (dtype_and_precision_registry as dtype_registry,
                             default_precision, datatype, NativeInteger,
                             NativeReal, NativeComplex, NativeBool, str_dtype)

from .literals       import LiteralInteger, LiteralFloat, LiteralComplex
from .literals       import LiteralTrue, LiteralFalse
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
# TODO [YG, 22.05.2020]: change properties to read _dtype, _prec, _rank, etc...
class NumpyArray(Application, NumpyNewArray):
    """
    Represents a call to  numpy.array for code generation.

    arg : list ,tuple ,Tuple, PythonList

    """

    def __new__(cls, arg, dtype=None, order='C'):

        if not isinstance(arg, (Tuple, PythonTuple, PythonList)):
            raise TypeError('Uknown type of  %s.' % type(arg))

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

        # Create instance, add attributes, and return it
        return Basic.__new__(cls, arg, dtype, order, prec)

    def __init__(self, arg, dtype=None, order='C'):
        arg_shape   = numpy.asarray(arg).shape
        self._shape = process_shape(arg_shape)
        self._rank  = len(self._shape)

    def _sympystr(self, printer):
        return self.arg

    @property
    def arg(self):
        return self._args[0]

    @property
    def dtype(self):
        return self._args[1]

    @property
    def order(self):
        return self._args[2]

    @property
    def precision(self):
        return self._args[3]

    @property
    def shape(self):
        return self._shape

    @property
    def rank(self):
        return self._rank

#==============================================================================
class NumpySum(Function, PyccelAstNode):
    """Represents a call to  numpy.sum for code generation.

    arg : list , tuple , PythonTuple, Tuple, PythonList, Variable
    """

    def __new__(cls, arg):
        if not isinstance(arg, (list, tuple, PythonTuple, Tuple, PythonList,
                            Variable, Expr)):
            raise TypeError('Uknown type of  %s.' % type(arg))

        return Basic.__new__(cls, arg)

    def __init__(self, arg):
        self._dtype = arg.dtype
        self._rank  = 0
        self._shape = ()
        self._precision = default_precision[str_dtype(self._dtype)]

    @property
    def arg(self):
        return self._args[0]

#==============================================================================
class NumpyProduct(Function, PyccelAstNode):
    """Represents a call to  numpy.prod for code generation.

    arg : list , tuple , PythonTuple, Tuple, PythonList, Variable
    """

    def __new__(cls, arg):
        if not isinstance(arg, (list, tuple, PythonTuple, Tuple, PythonList,
                                Variable, Expr)):
            raise TypeError('Uknown type of  %s.' % type(arg))
        return Basic.__new__(cls, arg)

    def __init__(self, arg):
        self._dtype = arg.dtype
        self._rank  = 0
        self._shape = ()
        self._precision = default_precision[str_dtype(self._dtype)]

    @property
    def arg(self):
        return self._args[0]


#==============================================================================
class NumpyMatmul(Application, PyccelAstNode):
    """Represents a call to numpy.matmul for code generation.
    arg : list , tuple , PythonTuple, Tuple, PythonList, Variable
    """

    def __new__(cls, a, b):
        if not isinstance(a, (list, tuple, PythonTuple, Tuple, PythonList,
                                Variable, Expr)):
            raise TypeError('Uknown type of  %s.' % type(a))
        if not isinstance(b, (list, tuple, PythonTuple, Tuple, PythonList,
                                Variable, Expr)):
            raise TypeError('Uknown type of  %s.' % type(a))
        return Basic.__new__(cls, a, b)

    def __init__(self, a ,b):

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
class NumpyLinspace(Application, NumpyNewArray):

    """
    Represents numpy.linspace.

    """

    def __new__(cls, *args):


        _valid_args = (Variable, IndexedElement, LiteralFloat,
                       sp_Integer, sp_Rational)

        for arg in args:
            if not isinstance(arg, _valid_args):
                raise TypeError('Expecting valid args')

        if len(args) == 3:
            start = args[0]
            stop = args[1]
            size = args[2]

        else:
            raise ValueError('Range has at most 3 arguments')

        index = Variable('int', 'linspace_index')
        return Basic.__new__(cls, start, stop, size, index)

    @property
    def start(self):
        return self._args[0]

    @property
    def stop(self):
        return self._args[1]

    @property
    def size(self):
        return self._args[2]

    @property
    def index(self):
        return self._args[3]

    @property
    def step(self):
        return (self.stop - self.start) / (self.size - 1)


    @property
    def dtype(self):
        return 'real'

    @property
    def order(self):
        return 'F'

    @property
    def precision(self):
        return default_precision['real']

    @property
    def shape(self):
        return (self.size,)

    @property
    def rank(self):
        return 1

    def _eval_is_real(self):
        return True

    def _sympystr(self, printer):
        sstr = printer.doprint
        code = 'linspace({}, {}, {})',format(sstr(self.start),
                                             sstr(self.stop),
                                             sstr(self.size))
        return code

#==============================================================================
class NumpyWhere(Application, NumpyNewArray):
    """ Represents a call to  numpy.where """

    def __new__(cls, mask):
        return Basic.__new__(cls, mask)


    @property
    def mask(self):
        return self._args[0]

    @property
    def index(self):
        ind = Variable('int','ind1')

        return ind

    @property
    def dtype(self):
        return 'int'

    @property
    def rank(self):
        return 2

    @property
    def shape(self):
        return ()

    @property
    def order(self):
        return 'F'

 #==============================================================================
class NumpyRand(Function, NumpyNewArray):

    """
      Represents a call to  numpy.random.random or numpy.random.rand for code generation.

    """
    _dtype = NativeReal()
    _precision = default_precision['real']

    def __init__(self, *args):
        self._shape = args
        self._rank  = len(self.shape)

    @property
    def order(self):
        return 'C'

#==============================================================================
class NumpyRandint(Function, NumpyNewArray):

    """
      Represents a call to  numpy.random.random or numpy.random.rand for code generation.

    """
    _dtype = NativeInteger()
    _precision = default_precision['integer']

    def __new__(cls, low, high = None, size = None):
        return Function.__new__(cls)

    def __init__(self, low, high = None, size = None):
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
    def order(self):
        return 'C'

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
class NumpyFull(Application, NumpyNewArray):
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
    def __new__(cls, shape, fill_value, dtype=None, order='C'):

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
        if stype != 'complex':
            from pyccel.codegen.printing.fcode import python_builtin_datatypes
            cast_func  = python_builtin_datatypes[stype]
            fill_value = cast_func(fill_value)

        return Basic.__new__(cls, shape, dtype, order, precision, fill_value)

    #--------------------------------------------------------------------------
    @property
    def shape(self):
        return self._args[0]

    @property
    def dtype(self):
        return self._args[1]

    @property
    def order(self):
        return self._args[2]

    @property
    def precision(self):
        return self._args[3]

    @property
    def fill_value(self):
        return self._args[4]

    @property
    def rank(self):
        return len(self.shape)

#==============================================================================
class NumpyAutoFill(NumpyFull):
    """ Abstract class for all classes which inherit from NumpyFull but
        the fill_value is implicitly specified
    """
    def __new__(cls, shape, dtype='float', order='C'):

        # Convert shape to PythonTuple
        shape = process_shape(shape)

        # Verify dtype and get precision
        dtype, precision = process_dtype(dtype)

        # Verify array ordering
        order = cls._process_order(order)

        return Basic.__new__(cls, shape, dtype, order, precision)
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
    # TODO [YG, 09.11.2020]: create LiteralInteger/LiteralFloat/LiteralComplex w/ correct precision
    @property
    def fill_value(self):
        dtype = self.dtype
        if isinstance(dtype, NativeInteger):
            value = LiteralInteger(0)
        elif isinstance(dtype, NativeReal):
            value = LiteralFloat(0)
        elif isinstance(dtype, NativeComplex):
            value = LiteralComplex(0., 0.)
        elif isinstance(dtype, NativeBool):
            value = LiteralFalse()
        else:
            raise TypeError('Unknown type')
        return value

#==============================================================================
class NumpyOnes(NumpyAutoFill):
    """ Represents a call to numpy.ones for code generation.
    """
    # TODO [YG, 09.11.2020]: create LiteralInteger/LiteralFloat/LiteralComplex w/ correct precision
    @property
    def fill_value(self):
        dtype = self.dtype
        if isinstance(dtype, NativeInteger):
            value = LiteralInteger(1)
        elif isinstance(dtype, NativeReal):
            value = LiteralFloat(1.)
        elif isinstance(dtype, NativeComplex):
            value = LiteralComplex(1., 0.)
        elif isinstance(dtype, NativeBool):
            value = LiteralTrue()
        else:
            raise TypeError('Unknown type')
        return value

#=======================================================================================
class NumpyFullLike(Application):
    """ Represents a call to numpy.full_like for code generation.
    """
    def __new__(cls, a, fill_value, dtype=None, order='K', subok=True):

        # NOTE: we ignore 'subok' argument
        dtype = a.dtype if (dtype is None) or isinstance(dtype, Nil) else dtype
        order = a.order if str(order).strip('\'"') in ('K', 'A') else order

        return NumpyFull(Shape(a), fill_value, dtype, order)

#=======================================================================================
class NumpyEmptyLike(Application):
    """ Represents a call to numpy.empty_like for code generation.
    """
    def __new__(cls, a, dtype=None, order='K', subok=True):

        # NOTE: we ignore 'subok' argument
        dtype = a.dtype if (dtype is None) or isinstance(dtype, Nil) else dtype
        order = a.order if str(order).strip('\'"') in ('K', 'A') else order

        return NumpyEmpty(Shape(a), dtype, order)

#=======================================================================================
class NumpyOnesLike(Application):
    """ Represents a call to numpy.ones_like for code generation.
    """
    def __new__(cls, a, dtype=None, order='K', subok=True):

        # NOTE: we ignore 'subok' argument
        dtype = a.dtype if (dtype is None) or isinstance(dtype, Nil) else dtype
        order = a.order if str(order).strip('\'"') in ('K', 'A') else order

        return NumpyOnes(Shape(a), dtype, order)

#=======================================================================================
class NumpyZerosLike(Application):
    """ Represents a call to numpy.zeros_like for code generation.
    """
    def __new__(cls, a, dtype=None, order='K', subok=True):

        # NOTE: we ignore 'subok' argument
        dtype = a.dtype if (dtype is None) or isinstance(dtype, Nil) else dtype
        order = a.order if str(order).strip('\'"') in ('K', 'A') else order

        return NumpyZeros(Shape(a), dtype, order)

#=======================================================================================

class NumpyNorm(Function, PyccelAstNode):
    """ Represents call to numpy.norm"""

    is_zero = False

    def __new__(cls, arg, dim=None):
        if isinstance(dim, ValuedArgument):
            dim = dim.value
        return Basic.__new__(cls, arg, dim)

    @property
    def arg(self):
        return self._args[0]

    @property
    def dim(self):
        return self._args[1]

    @property
    def dtype(self):
        return 'real'


    def shape(self, sh):
        if self.dim is not None:
            sh = list(sh)
            del sh[self.dim]
            return tuple(sh)
        else:
            return ()

    @property
    def rank(self):
        if self.dim is not None:
            return self.arg.rank-1
        return 0

#=====================================================
class Sqrt(PyccelPow):
    def __new__(cls, base):
        return PyccelPow(PyccelAssociativeParenthesis(base), LiteralFloat(0.5))

#====================================================


#==============================================================================
# Numpy universal functions
# https://numpy.org/doc/stable/reference/ufuncs.html#available-ufuncs
#
# NOTE: since we are subclassing sympy.Function, we need to use a name ending
#       with "Base", otherwise the Sympy's printer is going to skip this class.
#==============================================================================
class NumpyUfuncBase(Function, PyccelAstNode):
    """Base class for Numpy's universal functions."""

#------------------------------------------------------------------------------
class NumpyUfuncUnary(NumpyUfuncBase):
    """Numpy's universal function with one argument.
    """
    def __init__(self, x):
        self._shape      = x.shape
        self._rank       = x.rank
        self._dtype      = x.dtype if x.dtype is NativeComplex() else NativeReal()
        self._precision  = default_precision[str_dtype(self._dtype)]

#------------------------------------------------------------------------------
class NumpyUfuncBinary(NumpyUfuncBase):
    """Numpy's universal function with two arguments.
    """
    # TODO: apply Numpy's broadcasting rules to get shape/rank of output
    def __init__(self, x1, x2):
        self._shape     = x1.shape  # TODO ^^
        self._rank      = x1.rank   # TODO ^^
        self._dtype     = NativeReal()
        self._precision = default_precision['real']

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
    def __init__(self, x):
        self._shape     = x.shape
        self._rank      = x.rank
        self._dtype     = NativeInteger() if x.dtype is NativeInteger() else NativeReal()
        self._precision = default_precision[str_dtype(self._dtype)]
        self._order     = x.order


class NumpyFloor(NumpyUfuncUnary):
    def __init__(self, x):
        self._shape     = x.shape
        self._rank      = x.rank
        self._dtype     = NativeReal()
        self._precision = default_precision[str_dtype(self._dtype)]

class NumpyMod(NumpyUfuncBinary):
    def __init__(self, x1, x2):
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

class NumpyMin(NumpyUfuncUnary):
    def __init__(self, x):
        self._shape     = ()
        self._rank      = 0
        self._dtype     = x.dtype
        self._precision = x.precision

class NumpyMax(NumpyUfuncUnary):
    def __init__(self, x):
        self._shape     = ()
        self._rank      = 0
        self._dtype     = x.dtype
        self._precision = x.precision


#=======================================================================================
class NumpyComplex(PythonComplex):
    """ Represents a call to numpy.complex() function.
    """
    def __new__(cls, arg0, arg1=LiteralFloat(0)):
        return PythonComplex.__new__(cls, arg0, arg1)

class NumpyComplex64(NumpyComplex):
    _precision = dtype_registry['complex64'][1]

class NumpyComplex128(NumpyComplex):
    _precision = dtype_registry['complex128'][1]

#=======================================================================================
class NumpyFloat(PythonFloat):
    """ Represents a call to numpy.float() function.
    """
    def __new__(cls, arg):
        return PythonFloat.__new__(cls, arg)

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
