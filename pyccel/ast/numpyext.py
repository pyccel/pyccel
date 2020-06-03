#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy
from sympy import Basic, Function, Tuple
from sympy import Integer as sp_Integer
from sympy import Expr
from sympy import Rational as sp_Rational
from sympy import IndexedBase
from sympy.core.function import Application
from sympy.core.assumptions import StdFactKB
from sympy.logic.boolalg import BooleanTrue, BooleanFalse

from .basic import PyccelAstNode
from .core  import (Variable, IndexedElement, Slice, Len,
                   For, Range, Assign, List, Nil,
                   ValuedArgument, Constant, process_shape)

from .core           import PyccelPow, PyccelMinus, PyccelAssociativeParenthesis

from .builtins       import Int as PythonInt, Bool as PythonBool
from .builtins       import PythonFloat, PythonTuple, PythonComplex
from .datatypes      import dtype_and_precision_registry as dtype_registry
from .datatypes      import default_precision
from .datatypes      import datatype
from .datatypes      import NativeInteger, NativeReal, NativeComplex, NativeBool
from .numbers        import Integer, Float
from .type_inference import str_dtype


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
    'Cross',
    'Diag',
    'Empty',
    'EmptyLike',
    'NumpyFloat',
    'NumpyComplex',
    'Complex64',
    'Complex128',
    'Float32',
    'Float64',
    'Full',
    'FullLike',
    'Imag',
    'NumpyInt',
    'Int32',
    'Int64',
    'Linspace',
    'Matmul',
    'Max',
    'Min',
    'Mod',
    'Norm',
    'NumpySum',
    'Ones',
    'OnesLike',
    'Product',
    'PyccelArraySize',
    'Rand',
    'Real',
    'Shape',
    'Where',
    'Zeros',
    'ZerosLike'
)

#==============================================================================
numpy_constants = {
    'pi': Constant('real', 'pi', value=numpy.pi),
}

#==============================================================================
# TODO [YG, 18.02.2020]: accept Numpy array argument
# TODO [YG, 18.02.2020]: use order='K' as default, like in numpy.array
# TODO [YG, 22.05.2020]: move dtype & prec processing to __init__
# TODO [YG, 22.05.2020]: change properties to read _dtype, _prec, _rank, etc...
class Array(Application, PyccelAstNode):
    """
    Represents a call to  numpy.array for code generation.

    arg : list ,tuple ,Tuple, List

    """

    def __new__(cls, arg, dtype=None, order='C'):

        if not isinstance(arg, (Tuple, PythonTuple, List)):
            raise TypeError('Uknown type of  %s.' % type(arg))

        # Determine dtype and (if possible) precision
        if dtype is not None:
            if isinstance(dtype, ValuedArgument):
                dtype = dtype.value
            dtype = str(dtype).replace('\'', '')
            dtype, prec = dtype_registry[dtype]
        else:
            dtype = arg.dtype
            prec  = arg.precision

        # If necessary, use default precision
        if not prec:
            prec = default_precision[dtype]

        # Convert dtype from string to Singleton
        dtype = datatype(dtype)

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
        sstr = printer.doprint
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

    def fprint(self, printer, lhs):
        """Fortran print."""

        # Always transpose indices because Numpy initial values are given with
        # row-major ordering, while Fortran initial values are column-major
        shape = self.shape[::-1]

        shape_code = ', '.join('0:' + printer(PyccelMinus(i, Integer(1))) for i in shape)

        lhs_code = printer(lhs)
        code_alloc = 'allocate({0}({1}))'.format(lhs_code, shape_code)
        arg = self.arg
        if self.rank > 1:
            import functools
            import operator
            arg = functools.reduce(operator.concat, arg)
            init_value = 'reshape(' + printer(arg) + ', ' + printer(Tuple(*shape)) + ')'
        else:
            init_value = printer(arg)

        # If Numpy array is stored with column-major ordering, transpose values
        if self.order == 'F' and self.rank > 1:
            init_value = 'transpose({})'.format(init_value)

        code_init = '{0} = {1}'.format(lhs_code, init_value)
        code = '{0}\n{1}'.format(code_alloc, code_init)

        return code

#==============================================================================
class NumpySum(Function, PyccelAstNode):
    """Represents a call to  numpy.sum for code generation.

    arg : list , tuple , PythonTuple, Tuple, List, Variable
    """

    def __new__(cls, arg):
        if not isinstance(arg, (list, tuple, PythonTuple, Tuple, List, Variable, Expr)):
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


    def fprint(self, printer, lhs=None):
        """Fortran print."""

        rhs_code = printer(self.arg)
        if lhs:
            lhs_code = printer(lhs)
            return '{0} = sum({1})'.format(lhs_code, rhs_code)
        return 'sum({0})'.format(rhs_code)

#==============================================================================
class Product(Function, PyccelAstNode):
    """Represents a call to  numpy.prod for code generation.

    arg : list , tuple , PythonTuple, Tuple, List, Variable
    """

    def __new__(cls, arg):
        if not isinstance(arg, (list, tuple, PythonTuple, Tuple, List, Variable, Expr)):
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

    def fprint(self, printer, lhs=None):
        """Fortran print."""

        rhs_code = printer(self.arg)
        if lhs:
            lhs_code = printer(lhs)
            return '{0} = product({1})'.format(lhs_code, rhs_code)
        return 'product({0})'.format(rhs_code)

#==============================================================================
class Matmul(Application, PyccelAstNode):
    """Represents a call to numpy.matmul for code generation.
    arg : list , tuple , PythonTuple, Tuple, List, Variable
    """

    def __new__(cls, a, b):
        if not isinstance(a, (list, tuple, PythonTuple, Tuple, List, Variable, Expr)):
            raise TypeError('Uknown type of  %s.' % type(a))
        if not isinstance(b, (list, tuple, PythonTuple, Tuple, List, Variable, Expr)):
            raise TypeError('Uknown type of  %s.' % type(a))
        return Basic.__new__(cls, a, b)

    @property
    def a(self):
        return self._args[0]

    @property
    def b(self):
        return self._args[1]

    @property
    def dtype(self):
        return self._args[0].dtype

    @property
    def rank(self):
        return 1 # TODO: make this general

    def fprint(self, printer, lhs=None):
        """Fortran print."""
        a_code = printer(self.a)
        b_code = printer(self.b)

        if lhs:
            lhs_code = printer(lhs)

        if self.a.order and self.b.order:
            if self.a.order != self.b.order:
                raise NotImplementedError("Mixed order matmul not supported.")

        # Fortran ordering
        if self.a.order == 'F':
            if lhs:
                return '{0} = matmul({1},{2})'.format(lhs_code, a_code, b_code)
            return 'matmul({0},{1})'.format(a_code, b_code)

        # C ordering
        if lhs:
            return '{0} = matmul({2},{1})'.format(lhs_code, a_code, b_code)
        return 'matmul({1},{0})'.format(a_code, b_code)

#==============================================================================

class PyccelArraySize(Function, PyccelAstNode):
    def __new__(cls, arg, index):
        if not isinstance(arg, (list,
                                tuple,
                                Tuple,
                                PythonTuple,
                                List,
                                Array,
                                Variable,
                                IndexedElement,
                                IndexedBase)):
            raise TypeError('Uknown type of  %s.' % type(arg))

        return Basic.__new__(cls, arg, index)

    def __init__(self, arg, index):
        self._dtype = NativeInteger()
        self._rank  = 0
        self._shape = ()
        self._precision = default_precision['integer']

    @property
    def arg(self):
        return self._args[0]

    @property
    def index(self):
        return self._args[1]

    def _sympystr(self, printer):
        return 'Shape({},{})'.format(str(self.arg), str(self.index))

    def fprint(self, printer, lhs = None):
        """Fortran print."""

        lhs_code = printer(lhs)
        if isinstance(self.arg, Array):
            init_value = printer(self.arg.arg)
        else:
            init_value = printer(self.arg)

        if self.arg.order == 'C':
            index = printer(self.arg.rank - self.index)
        else:
            index = printer(self.index + 1)

        if lhs:
            code_init = '{0} = size({1}, {2})'.format(lhs_code, init_value, index)
        else:
            code_init = 'size({0}, {1})'.format(init_value, index)

        return code_init

def Shape(arg):
    if arg.shape is None:
        return PythonTuple(*(PyccelArraySize(arg,i) for i in range(arg.rank)))
    elif isinstance(arg.shape, PythonTuple):
        return arg.shape
    else:
        return PythonTuple(*arg.shape)

#==============================================================================
# TODO [YG, 09.03.2020]: Reconsider this class, given new ast.builtins.Float
class Real(Function, PyccelAstNode):

    """Represents a call to  numpy.real for code generation.

    arg : Variable, Float, sp_Integer, Complex
    """

    def __new__(cls, arg):

        _valid_args = (Variable, IndexedElement, sp_Integer, Nil,
                       Float, Expr, Application)

        if not isinstance(arg, _valid_args):
            raise TypeError('Uknown type of  %s.' % type(arg))
        return Basic.__new__(cls, arg)

    def __init__(self, arg):
        self._dtype = NativeReal()
        self._rank  = 0
        self._shape = ()
        self._precision = default_precision['real']

    @property
    def arg(self):
        return self._args[0]

    def fprint(self, printer):
        """Fortran print."""

        value = printer(self.arg)
        prec  = printer(self.precision)
        code = 'Real({0}, {1})'.format(value, prec)
        return code


    def __str__(self):
        return 'Float({0})'.format(str(self.arg))


    def _sympystr(self, printer):
        return self.__str__()

#==============================================================================
class Imag(Real):

    """Represents a call to  numpy.imag for code generation.

    arg : Variable, Float, sp_Integer, Complex
    """

    def fprint(self, printer):
        """Fortran print."""

        value = printer(self.arg)
        code = 'aimag({0})'.format(value)
        return code


    def __str__(self):
        return 'imag({0})'.format(str(self.arg))

#==============================================================================
class Linspace(Application, PyccelAstNode):

    """
    Represents numpy.linspace.

    """

    def __new__(cls, *args):


        _valid_args = (Variable, IndexedElement, Float,
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


    def fprint(self, printer, lhs=None):
        """Fortran print."""

        init_value = '[({0} + {1}*{2},{1} = 0,{3}-1)]'

        start = printer(self.start)
        step  = printer(self.step)
        stop  = printer(self.stop)
        index = printer(self.index)

        init_value = init_value.format(start, index, step, stop)



        if lhs:
            lhs    = printer(lhs)
            code   = 'allocate(0:{})'.format(printer(self.size))
            code  += '\n{0} = {1}'.format(lhs, init_value)
        else:
            code   = '{0}'.format(init_value)

        return code

#==============================================================================
class Diag(Application, PyccelAstNode):

    """
    Represents numpy.diag.

    """
    #TODO improve the properties dtype, rank, shape
    # to be more general


    def __new__(cls, array, v=0, k=0):


        _valid_args = (Variable, IndexedElement, PythonTuple, Tuple)


        if not isinstance(array, _valid_args):
           raise TypeError('Expecting valid args')

        if not isinstance(k, (int, sp_Integer)):
           raise ValueError('k must be an integer')

        index = Variable('int', 'diag_index')
        return Basic.__new__(cls, array, v, k, index)

    @property
    def array(self):
        return self._args[0]

    @property
    def v(self):
        return self._args[1]

    @property
    def k(self):
        return self._args[2]

    @property
    def index(self):
        return self._args[3]


    @property
    def dtype(self):
        return 'real'

    @property
    def order(self):
        return 'C'

    @property
    def precision(self):
        return default_precision['real']

    @property
    def shape(self):
        return Len(self.array)

    @property
    def rank(self):
        rank = 1 if self.array.rank == 2 else 2
        return rank


    def fprint(self, printer, lhs):
        """Fortran print."""

        array = printer(self.array)
        rank  = self.array.rank
        index = printer(self.index)

        if rank == 2:
            lhs   = IndexedBase(lhs)[self.index]
            rhs   = IndexedBase(self.array)[self.index,self.index]
            body  = [Assign(lhs, rhs)]
            body  = For(self.index, Range(Len(self.array)), body)
            code  = printer(body)
            alloc = 'allocate({0}(0: size({1},1)-1))'.format(lhs.base, array)
        elif rank == 1:

            lhs   = IndexedBase(lhs)[self.index, self.index]
            rhs   = IndexedBase(self.array)[self.index]
            body  = [Assign(lhs, rhs)]
            body  = For(self.index, Range(Len(self.array)), body)
            code  = printer(body)
            alloc = 'allocate({0}(0: size({1},1)-1, 0: size({1},1)-1))'.format(lhs, array)

        return alloc + '\n' + code

#==============================================================================
class Cross(Application, PyccelAstNode):

    """
    Represents numpy.cross.

    """
    #TODO improve the properties dtype, rank, shape
    # to be more general

    def __new__(cls, a, b):


        _valid_args = (Variable, IndexedElement, PythonTuple, Tuple)


        if not isinstance(a, _valid_args):
           raise TypeError('Expecting valid args')

        if not isinstance(b, _valid_args):
           raise TypeError('Expecting valid args')

        return Basic.__new__(cls, a, b)

    @property
    def first(self):
        return self._args[0]

    @property
    def second(self):
        return self._args[1]


    @property
    def dtype(self):
        return self.first.dtype

    @property
    def order(self):
        #TODO which order should we give it
        return 'C'

    @property
    def precision(self):
        return self.first.precision


    @property
    def rank(self):
        return self.first.rank

    @property
    def shape(self):
        return ()


    def fprint(self, printer, lhs=None):
        """Fortran print."""

        a     = IndexedBase(self.first)
        b     = IndexedBase(self.second)
        slc   = Slice(None, None)
        rank  = self.rank

        if rank > 2:
            raise NotImplementedError('TODO')

        if rank == 2:
            a_inds = [[slc,0], [slc,1], [slc,2]]
            b_inds = [[slc,0], [slc,1], [slc,2]]

            if self.first.order == 'C':
                for inds in a_inds:
                    inds.reverse()
            if self.second.order == 'C':
                for inds in b_inds:
                    inds.reverse()

            a = [a[tuple(inds)] for inds in a_inds]
            b = [b[tuple(inds)] for inds in b_inds]


        cross_product = [a[1]*b[2]-a[2]*b[1],
                         a[2]*b[0]-a[0]*b[2],
                         a[0]*b[1]-a[1]*b[0]]

        cross_product = PythonTuple(cross_product)
        cross_product = printer(cross_product)
        first = printer(self.first)
        order = self.order

        if lhs is not None:
            lhs  = printer(lhs)

            if rank == 2:
                alloc = 'allocate({0}(0:size({1},1)-1,0:size({1},2)-1))'.format(lhs, first)

            elif rank == 1:
                alloc = 'allocate({}(0:size({})-1)'.format(lhs, first)



        if rank == 2:

            if order == 'C':

                code = 'reshape({}, shape({}), order=[2, 1])'.format(cross_product, first)
            else:

                code = 'reshape({}, shape({})'.format(cross_product, first)

        elif rank == 1:
            code = cross_product

        if lhs is not None:
            code = '{} = {}'.format(lhs, code)

        #return alloc + '\n' + code
        return code

#==============================================================================
class Where(Application, PyccelAstNode):
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


    def fprint(self, printer, lhs):

        ind   = printer(self.index)
        mask  = printer(self.mask)
        lhs   = printer(lhs)

        stmt  = 'pack([({ind},{ind}=0,size({mask})-1)],{mask})'.format(ind=ind,mask=mask)
        stmt  = '{lhs}(:,0) = {stmt}'.format(lhs=lhs, stmt=stmt)
        alloc = 'allocate({}(0:count({})-1,0:0))'.format(lhs, mask)

        return alloc +'\n' + stmt

#==============================================================================
class Rand(Function, PyccelAstNode):

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

    def fprint(self, printer, lhs, stack_array=False):
        """Fortran print."""

        lhs_code = printer(lhs)
        stmts = []

        if self.rank>0:
            # Create statement for allocation
            if not stack_array:
                # Transpose indices because of Fortran column-major ordering
                shape = self.shape[::-1]

                shape_code = ', '.join('0:' + printer(PyccelMinus(i, Integer(1))) for i in shape)

                code_alloc = 'allocate({0}({1}))'.format(lhs_code, shape_code)
                stmts.append(code_alloc)

        # Create statement for initialization
        code_init = 'call random_number({0})'.format(lhs_code)
        stmts.append(code_init)

        return '\n'.join(stmts)


#==============================================================================
class Full(Application, PyccelAstNode):
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
        dtype, precision = cls._process_dtype(dtype)

        # Verify array ordering
        order = cls._process_order(order)

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

    #--------------------------------------------------------------------------
    @staticmethod
    def _process_dtype(dtype):
        if dtype  in (PythonInt, PythonFloat, PythonComplex, PythonBool, NumpyInt, 
                      Int32, Int64, NumpyComplex, Complex64, Complex128, NumpyFloat,
                      Float64, Float32):
            dtype = dtype.__name__.lower()
        else:
            dtype            = str(dtype).replace('\'', '').lower()
        dtype, precision = dtype_registry[dtype]
        dtype            = datatype(dtype)

        return dtype, precision

    #--------------------------------------------------------------------------
    @staticmethod
    def _process_order(order):

        if (order is None) or isinstance(order, Nil):
            return None

        order = str(order).strip('\'"')
        if order not in ('C', 'F'):
            raise ValueError('unrecognized order = {}'.format(order))
        return order

    #--------------------------------------------------------------------------
    def fprint(self, printer, lhs, stack_array=False):
        """Fortran print."""

        lhs_code = printer(lhs)
        stmts = []

        # Create statement for allocation
        if not stack_array:
            # Transpose indices because of Fortran column-major ordering
            shape = self.shape if self.order == 'F' else self.shape[::-1]

            shape_code = ', '.join('0:' + printer(PyccelMinus(i, Integer(1))) for i in shape)

            code_alloc = 'allocate({0}({1}))'.format(lhs_code, shape_code)
            stmts.append(code_alloc)

        # Create statement for initialization
        if self.fill_value is not None:
            init_value = printer(self.fill_value)
            code_init = '{0} = {1}'.format(lhs_code, init_value)
            stmts.append(code_init)

        return '\n'.join(stmts)

#==============================================================================
class Empty(Full):
    """ Represents a call to numpy.empty for code generation.
    """
    def __new__(cls, shape, dtype='float', order='C'):

        # Convert shape to PythonTuple
        shape = process_shape(shape)

        # Verify dtype and get precision
        dtype, precision = cls._process_dtype(dtype)

        # Verify array ordering
        order = cls._process_order(order)

        return Basic.__new__(cls, shape, dtype, order, precision)

    @property
    def fill_value(self):
        return None

#==============================================================================
class Zeros(Empty):
    """ Represents a call to numpy.zeros for code generation.
    """
    @property
    def fill_value(self):
        dtype = self.dtype
        if isinstance(dtype, NativeInteger):
            value = 0
        elif isinstance(dtype, NativeReal):
            value = 0.0
        elif isinstance(dtype, NativeComplex):
            value = 0.0
        elif isinstance(dtype, NativeBool):
            value = BooleanFalse()
        else:
            raise TypeError('Unknown type')
        return value

#==============================================================================
class Ones(Empty):
    """ Represents a call to numpy.ones for code generation.
    """
    @property
    def fill_value(self):
        dtype = self.dtype
        if isinstance(dtype, NativeInteger):
            value = 1
        elif isinstance(dtype, NativeReal):
            value = 1.0
        elif isinstance(dtype, NativeComplex):
            value = 1.0
        elif isinstance(dtype, NativeBool):
            value = BooleanTrue()
        else:
            raise TypeError('Unknown type')
        return value

#=======================================================================================
class FullLike(Application):

    def __new__(cls, a, fill_value, dtype=None, order='K', subok=True):

        # NOTE: we ignore 'subok' argument
        dtype = a.dtype if (dtype is None) or isinstance(dtype, Nil) else dtype
        order = a.order if str(order).strip('\'"') in ('K', 'A') else order

        return Full(Shape(a), fill_value, dtype, order)

#=======================================================================================
class EmptyLike(Application):

    def __new__(cls, a, dtype=None, order='K', subok=True):

        # NOTE: we ignore 'subok' argument
        dtype = a.dtype if (dtype is None) or isinstance(dtype, Nil) else dtype
        order = a.order if str(order).strip('\'"') in ('K', 'A') else order

        return Empty(Shape(a), dtype, order)

#=======================================================================================
class OnesLike(Application):

    def __new__(cls, a, dtype=None, order='K', subok=True):

        # NOTE: we ignore 'subok' argument
        dtype = a.dtype if (dtype is None) or isinstance(dtype, Nil) else dtype
        order = a.order if str(order).strip('\'"') in ('K', 'A') else order

        return Ones(Shape(a), dtype, order)

#=======================================================================================
class ZerosLike(Application):

    def __new__(cls, a, dtype=None, order='K', subok=True):

        # NOTE: we ignore 'subok' argument
        dtype = a.dtype if (dtype is None) or isinstance(dtype, Nil) else dtype
        order = a.order if str(order).strip('\'"') in ('K', 'A') else order

        return Zeros(Shape(a), dtype, order)

#=======================================================================================

class Norm(Function, PyccelAstNode):
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

    def fprint(self, printer):
        """Fortran print."""

        if self.dim:
            rhs = 'Norm2({},{})'.format(printer(self.arg),printer(self.dim))
        else:
            rhs = 'Norm2({})'.format(printer(self.arg))

        return rhs

#=====================================================
class Sqrt(PyccelPow):
    def __new__(cls, base):
        return PyccelPow(PyccelAssociativeParenthesis(base), Float(0.5))

#====================================================

class Mod(Function, PyccelAstNode):
    def __new__(cls,*args):
        return Basic.__new__(cls, *args)

    def __init__(self,*args):
        assumptions={'integer':True}
        ass_copy = assumptions.copy()
        self._assumptions = StdFactKB(assumptions)
        self._assumptions._generator = ass_copy

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
        self._shape     = x.shape
        self._rank      = x.rank
        self._dtype     = x.dtype if x.dtype is NativeComplex() else NativeReal()
        self._precision = default_precision[str_dtype(self._dtype)]

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
#class NumpyFabs    (NumpyUfuncUnary): pass
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


class NumpyFloor(NumpyUfuncUnary):
    def __init__(self, x):
        self._shape     = x.shape
        self._rank      = x.rank
        self._dtype     = NativeReal()
        self._precision = default_precision[str_dtype(self._dtype)]


class Min(Function, PyccelAstNode):
    def _eval_is_integer(self):
        return all(i.is_integer for i in self.args)

class Max(Function, PyccelAstNode):
    def _eval_is_integer(self):
        return all(i.is_integer for i in self.args)


#=======================================================================================
class NumpyComplex(PythonComplex):
    """ Represents a call to numpy.complex() function.
    """
    def __new__(cls, arg0, arg1=Float(0)):
        return PythonComplex.__new__(cls, arg0, arg1)

class Complex64(NumpyComplex):
    _precision = dtype_registry['complex64'][1]

class Complex128(NumpyComplex):
    _precision = dtype_registry['complex128'][1]

#=======================================================================================
class NumpyFloat(PythonFloat):
    """ Represents a call to numpy.float() function.
    """
    def __new__(cls, arg):
        return PythonFloat.__new__(cls, arg)

class Float32(NumpyFloat):
    """ Represents a call to numpy.float32() function.
    """
    _precision = dtype_registry['float32'][1]

class Float64(NumpyFloat):
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

class Int32(NumpyInt):
    """ Represents a call to numpy.int32() function.
    """
    _precision = dtype_registry['int32'][1]

class Int64(NumpyInt):
    """ Represents a call to numpy.int64() function.
    """
    _precision = dtype_registry['int64'][1]


