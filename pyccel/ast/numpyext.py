#!/usr/bin/python
# -*- coding: utf-8 -*-

# TODO remove sympify, Symbol

import numpy
from sympy.core.function import Function
from sympy.core import Symbol, Tuple
from sympy import sympify
from sympy.core.basic import Basic
from sympy import Integer as sp_Integer, Add, Mul, Pow, Float as sp_Float
from sympy.utilities.iterables import iterable
from sympy.logic.boolalg import Boolean, BooleanTrue, BooleanFalse
from sympy.core.assumptions import StdFactKB
from sympy import sqrt, asin, acsc, acos, asec, atan, acot, sinh, cosh, tanh, log
from sympy import Rational as sp_Rational
from sympy import IndexedBase


from .core import (Variable, IndexedElement, IndexedVariable, List, String, ValuedArgument,Constant)
from .datatypes import dtype_and_precsision_registry as dtype_registry
from .datatypes import default_precision
from .datatypes import DataType, datatype
from .datatypes import (NativeInteger, NativeReal, NativeComplex,
                        NativeBool)

from .core import local_sympify ,float2int

numpy_constants = {
    'pi': Constant('real', 'pi', value=numpy.pi),
                  }

class Array(Function):

    """Represents a call to  numpy.array for code generation.

    arg : list ,tuple ,Tuple,List
    """

    def __new__(cls, arg, dtype=None):
        if not isinstance(arg, (list, tuple, Tuple, List)):
            raise TypeError('Uknown type of  %s.' % type(arg))

        prec = 0
        
        if not dtype is None:
            if isinstance(dtype, ValuedArgument):
                dtype = dtype.value
            dtype = str(dtype).replace('\'', '')
            arg = float2int(arg) if 'int' in dtype else arg
                
            dtype,prec = dtype_registry[dtype]
            dtype = datatype('ndarray' + dtype)

        if not prec and dtype:
            prec = default_precision[dtype]
      
        return Basic.__new__(cls, arg, dtype, prec)

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
    def precision(self):
        return self._args[2]

    @property
    def shape(self):
        return Tuple(*numpy.shape(self.arg))

    @property
    def rank(self):
        return len(self.shape)

    def fprint(self, printer, lhs):
        """Fortran print."""

        if isinstance(self.shape, (Tuple, tuple)):

            # this is a correction. problem on LRZ

            shape_code = ', '.join('0:' + printer(i - 1) for i in
                                   self.shape)
        else:
            shape_code = '0:' + printer(self.shape - 1)
        lhs_code = printer(lhs)
        code_alloc = 'allocate({0}({1}))'.format(lhs_code, shape_code)
        arg = self.arg
        if self.rank > 1:
            import functools
            import operator
            arg = functools.reduce(operator.concat, arg)
            init_value = 'reshape(' + printer(arg) + ',' \
                + printer(self.shape) + ')'
        else:
            init_value = printer(arg)
        code_init = '{0} = {1}'.format(lhs_code, init_value)
        code = '{0}\n{1}'.format(code_alloc, code_init)
        return code


class Sum(Function):
    """Represents a call to  numpy.sum for code generation.

    arg : list , tuple , Tuple, List, Variable
    """

    def __new__(cls, arg):
        if not isinstance(arg, (list, tuple, Tuple, List, Variable)):
            raise TypeError('Uknown type of  %s.' % type(arg))
        return Basic.__new__(cls, arg)

    @property
    def arg(self):
        return self._args[0]

    @property
    def dtype(self):
        return self._args[0].dtype

    @property
    def rank(self):
        return 0

    def fprint(self, printer, lhs=None):
        """Fortran print."""

        rhs_code = printer(self.arg)
        if lhs:
            lhs_code = printer(lhs)
            return '{0} = sum({1})'.format(lhs_code, rhs_code)
        return 'sum({0})'.format(rhs_code)




class Shape(Array):

    """Represents a call to  numpy.shape for code generation.

    arg : list ,tuple ,Tuple,List, Variable
    """

    def __new__(cls, arg, index=None):
        if not isinstance(arg, (list,
                                tuple,
                                Tuple,
                                List,
                                Array,
                                Variable,
                                IndexedElement,
                                IndexedBase)):
            raise TypeError('Uknown type of  %s.' % type(arg))

        # TODO add check on index: must be Integer or Variable with dtype=int

        return Basic.__new__(cls, arg, index)

    @property
    def arg(self):
        return self._args[0]

    @property
    def index(self):
        return self._args[1]

    @property
    def dtype(self):
        return 'ndarrayint'

    @property
    def shape(self):
        return Tuple(self.arg.rank)

    @property
    def rank(self):
        return 1

    def fprint(self, printer, lhs = None):
        """Fortran print."""

        lhs_code = printer(lhs)
        if isinstance(self.arg, Array):
            init_value = printer(self.arg.arg)
        else:
            init_value = printer(self.arg)

        if lhs:
            if self.index is None:
                code_init = '{0} = shape({1})'.format(lhs_code, init_value)

            else:
                code_init = '{0} = size({1}, {2})'.format(lhs_code, init_value,
                                                          printer(self.index))

        else:
            if self.index is None:
                code_init = 'shape({0})'.format(init_value)

            else:
                code_init = 'size({0}, {1})'.format(init_value, printer(self.index))

        return code_init


class Int(Function):

    """Represents a call to  numpy.int for code generation.

    arg : Variable, Real, Integer, Complex
    """

    def __new__(cls, arg):
        if not isinstance(arg, (Variable, sp_Float, sp_Integer, Mul, Add, Pow, sp_Rational)):

            raise TypeError('Uknown type of  %s.' % type(arg))
        obj = Basic.__new__(cls, arg)
        assumptions = {'integer':True}
        ass_copy = assumptions.copy()
        obj._assumptions = StdFactKB(assumptions)
        obj._assumptions._generator = ass_copy
        return obj

    @property
    def arg(self):
        return self._args[0]

    @property
    def dtype(self):
        return 'int'

    @property
    def shape(self):
        return None

    @property
    def rank(self):
        return 0

    @property
    def precision(self):
        return 4


    def fprint(self, printer):
        """Fortran print."""

        value = printer(self.arg)
        code = 'Int({0})'.format(value)
        return code

class Real(Function):

    """Represents a call to  numpy.Real for code generation.

    arg : Variable, Float, Integer, Complex
    """

    def __new__(cls, arg):
        if not isinstance(arg, (Variable, sp_Integer, sp_Float, Mul, Add, Pow, sp_Rational)):
            raise TypeError('Uknown type of  %s.' % type(arg))
        obj = Basic.__new__(cls, arg)
        assumptions = {'real':True}
        ass_copy = assumptions.copy()
        obj._assumptions = StdFactKB(assumptions)
        obj._assumptions._generator = ass_copy
        return obj

    @property
    def arg(self):
        return self._args[0]

    @property
    def dtype(self):
        return 'int'

    @property
    def shape(self):
        return None

    @property
    def rank(self):
        return 0

    @property
    def precision(self):
        return 8

    def fprint(self, printer):
        """Fortran print."""

        value = printer(self.arg)
        code = 'Real({0})'.format(value)
        return code


    def __str__(self):
        return 'Float({0})'.format(str(self.arg))


    def _sympystr(self, printer):

        return self.__str__()

class Complex(Function):

    """Represents a call to  numpy.complex for code generation.

    arg : Variable, Float, Integer
    """

    def __new__(cls, arg0, arg1=sp_Float(0)):

        for arg in [arg0, arg1]:
            if not isinstance(arg, (Variable, sp_Integer, sp_Float, Mul, Add, Pow, sp_Rational)):
                raise TypeError('Uknown type of  %s.' % type(arg))
        obj = Basic.__new__(cls, arg0, arg1)
        assumptions = {'complex':True}
        ass_copy = assumptions.copy()
        obj._assumptions = StdFactKB(assumptions)
        obj._assumptions._generator = ass_copy
        return obj

    @property
    def real_part(self):
        return self._args[0]

    @property
    def imag_part(self):
        return self._args[1]

    @property
    def dtype(self):
        return 'complex'

    @property
    def shape(self):
        return None

    @property
    def rank(self):
        return 0

    @property
    def precision(self):
        return 8

    def fprint(self, printer):
        """Fortran print."""

        value0 = printer(self.real_part)
        value1 = printer(self.imag_part)
        code = 'complex({0},{1})'.format(value0,value1)
        return code


    def __str__(self):
        return self.fprint(str)


    def _sympystr(self, printer):

        return self.fprint(str)


class Rand(Real):

    """Represents a call to  numpy.random.random or numpy.random.rand for code generation.

    arg : list ,tuple ,Tuple,List
    """

    @property
    def arg(self):
        return self._args[0]

    @property
    def rank(self):
        return 0

    def fprint(self, printer):
        """Fortran print."""

        rhs_code = printer(self.arg)
        if len(self.arg) == 0:
            rhs_code = ''
        return 'rand({0})'.format(rhs_code)



class Zeros(Function):

    """Represents a call to numpy.zeros for code generation.

    shape : int, list, tuple
        int or list of integers

    dtype: str, DataType
        datatype for the constructed array

    Examples

    """

    # TODO improve

    def __new__(cls, shape,*args):

        args = list(args)
        args_= {'dtype':'real','order':'C'}
        prec = 0
        val_args = []

        for i in range(len(args)):
            if isinstance(args[i], ValuedArgument):
                val_args = args[i:]
                args[i:] = []
                break


        if len(args)==1:
            args_['dtype'] = str(args[0])
        elif len(args)==2:
            args_['dtype'] = str(args[0])
            args_['order'] = str(args[1])

        for i in val_args:
            val = str(i.value).replace('\'', '')
            args_[str(i.argument.name)] = val


        dtype = args_['dtype']
        order = args_['order']

        if isinstance(shape,Tuple):
            shape = list(shape)

        if isinstance(shape, list):
            if order == 'C':
                shape.reverse()

            # this is a correction. otherwise it is not working on LRZ
            if isinstance(shape[0], list):
                shape = Tuple(*(sympify(i, locals = local_sympify) for i in shape[0]))
            else:
                shape = Tuple(*(sympify(i, locals = local_sympify) for i in shape))

        elif isinstance(shape, (int, sp_Integer, Symbol)):
            shape = Tuple(sympify(shape, locals = local_sympify))
        else:
            shape = shape

        if isinstance(dtype, str):
            dtype = dtype.replace('\'', '')
            dtype, prec = dtype_registry[dtype]
            dtype = datatype('ndarray' + dtype)
        elif not isinstance(dtype, DataType):
            raise TypeError('datatype must be an instance of DataType.')

        if not prec:
            prec = default_precision[str(dtype)]

        return Basic.__new__(cls, shape, dtype, order, prec)

    @property
    def shape(self):
        return self._args[0]

    @property
    def rank(self):
        if iterable(self.shape):
            return len(self.shape)
        else:
            return 1

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
    def init_value(self):
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

    def fprint(self, printer, lhs):
        """Fortran print."""

        if isinstance(self.shape, (Tuple,tuple)):

            # this is a correction. problem on LRZ

            shape_code = ', '.join('0:' + printer(i - 1) for i in
                                   self.shape)
        else:

            shape_code = '0:' + printer(self.shape - 1)

        init_value = printer(self.init_value)

        lhs_code = printer(lhs)

        code_alloc = 'allocate({0}({1}))'.format(lhs_code, shape_code)
        code_init = '{0} = {1}'.format(lhs_code, init_value)
        code = '{0}\n{1}'.format(code_alloc, code_init)
        return code


class Ones(Zeros):

    """Represents a call to numpy.ones for code generation.

    shape : int or list of integers

    """

    @property
    def init_value(self):
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

class Empty(Zeros):

    """Represents a call to numpy.empty for code generation.

    shape : int or list of integers

    """
    def fprint(self, printer, lhs):
        """Fortran print."""

        if isinstance(self.shape, Tuple):

            # this is a correction. problem on LRZ

            shape_code = ', '.join('0:' + printer(i - 1) for i in
                                   self.shape)
        else:
            shape_code = '0:' + printer(self.shape - 1)


        lhs_code = printer(lhs)

        code = 'allocate({0}({1}))'.format(lhs_code, shape_code)
        return code



class Sqrt(Function):
    def __new__(cls,arg):
        obj = sqrt(arg)
        if arg.is_real:
            assumptions={'real':True}
            ass_copy = assumptions.copy()
            obj._assumptions = StdFactKB(assumptions)
            obj._assumptions._generator = ass_copy
        return obj

class Asin(Function):
    def __new__(cls,arg):
        obj = asin(arg)
        if arg.is_real:
            assumptions={'real':True}
            ass_copy = assumptions.copy()
            obj._assumptions = StdFactKB(assumptions)
            obj._assumptions._generator = ass_copy
        return obj

class Acos(Function):
    def __new__(cls,arg):
        obj = acos(arg)
        if arg.is_real:
            assumptions={'real':True}
            ass_copy = assumptions.copy()
            obj._assumptions = StdFactKB(assumptions)
            obj._assumptions._generator = ass_copy
        return obj

class Asec(Function):
    def __new__(cls,arg):
        obj = asec(arg)
        if arg.is_real:
            assumptions={'real':True}
            ass_copy = assumptions.copy()
            obj._assumptions = StdFactKB(assumptions)
            obj._assumptions._generator = ass_copy
        return obj

class Atan(Function):
    def __new__(cls,arg):
        obj = atan(arg)
        if arg.is_real:
            assumptions={'real':True}
            ass_copy = assumptions.copy()
            obj._assumptions = StdFactKB(assumptions)
            obj._assumptions._generator = ass_copy
        return obj

class Acot(Function):
    def __new__(cls,arg):
        obj = acot(arg)
        if arg.is_real:
            assumptions={'real':True}
            ass_copy = assumptions.copy()
            obj._assumptions = StdFactKB(assumptions)
            obj._assumptions._generator = ass_copy
        return obj


class Acsc(Function):
    def __new__(cls,arg):
        obj = acsc(arg)
        if arg.is_real:
            assumptions={'real':True}
            ass_copy = assumptions.copy()
            obj._assumptions = StdFactKB(assumptions)
            obj._assumptions._generator = ass_copy
        return obj

class Sinh(Function):
    def __new__(cls,arg):
        obj = sinh(arg)
        if arg.is_real:
            assumptions={'real':True}
            ass_copy = assumptions.copy()
            obj._assumptions = StdFactKB(assumptions)
            obj._assumptions._generator = ass_copy
        return obj

class Cosh(Function):
    def __new__(cls,arg):
        obj = cosh(arg)
        if arg.is_real:
            assumptions={'real':True}
            ass_copy = assumptions.copy()
            obj._assumptions = StdFactKB(assumptions)
            obj._assumptions._generator = ass_copy
        return obj

class Tanh(Function):
    def __new__(cls,arg):
        obj = tanh(arg)
        if arg.is_real:
            assumptions={'real':True}
            ass_copy = assumptions.copy()
            obj._assumptions = StdFactKB(assumptions)
            obj._assumptions._generator = ass_copy
        return obj

class Log(Function):
    def __new__(cls,arg):
        obj = log(arg)
        if arg.is_real:
            assumptions={'real':True}
            ass_copy = assumptions.copy()
            obj._assumptions = StdFactKB(assumptions)
            obj._assumptions._generator = ass_copy
        return obj


class Complex64(Complex):
    @property
    def precision(self):
        return 4

class Complex128(Complex):
    pass

class Float32(Real):
    @property
    def precision(self):
        return 4

class Float64(Real):
    pass

class Int32(Int):
    pass

class Int64(Int):
    @property
    def precision(self):
        return 8
