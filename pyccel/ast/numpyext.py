# coding: utf-8

# TODO remove sympify, Symbol
import numpy
from sympy.core.function import Function
from sympy.core import Symbol, Tuple
from sympy import sympify
from sympy.core.basic import Basic
from sympy.utilities.iterables import iterable
from sympy.logic.boolalg import Boolean, BooleanTrue, BooleanFalse

from .core import Variable, IndexedElement, IndexedVariable,List
from .core import DataType, datatype
from .core import (NativeInteger, NativeFloat, NativeDouble, NativeComplex,
                   NativeBool, String)


class Array(Function):
    """Represents a call to  numpy.array for code generation.

    ls : list ,tuple ,Tuple,List
    """
    def __new__(cls, arg, dtype=None):
        if not isinstance(arg,(list,tuple,Tuple,List) ):
            raise TypeError("Uknown type of  %s." % type(arg))
        if isinstance(dtype, str):
            dtype = datatype('ndarray'+dtype)

        return Basic.__new__(cls, arg, dtype)

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
    def shape(self):
        return Tuple(*numpy.shape(self._args[0]))

    @property
    def rank(self):
        return len(self.shape) 
    
    def fprint(self, printer, lhs):
        """Fortran print."""
        if isinstance(self.shape, (Tuple,tuple)):
        # this is a correction. problem on LRZ
            shape_code = ', '.join('0:' + printer(i-1) for i in self.shape)
        else:
            shape_code = '0:' + printer(self.shape-1)
        lhs_code = printer(lhs)

        code_alloc = "allocate({0}({1}))".format(lhs_code, shape_code)
        init_value = printer(self.arg)
        code_init = "{0} = {1}".format(lhs_code, init_value)
        code = "{0}\n{1}".format(code_alloc, code_init)

        return code

class Sum(Function):
    def __new__(cls, arg):
        if not isinstance(arg,(list,tuple,Tuple,List, Variable) ):
            raise TypeError("Uknown type of  %s." % type(arg))
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
    def __new__(cls, arg):
        if not isinstance(arg,(list,tuple,Tuple,List,Array, Variable) ):
            raise TypeError("Uknown type of  %s." % type(arg))
        return Basic.__new__(cls, arg)

    @property
    def arg(self):
        return self._args[0]
    
    @property
    def dtype(self):
        return 'ndarrayint'
    
    @property
    def shape(self):
        return Tuple(self.arg.rank,)

    @property
    def rank(self):
        return 1 
  
    def fprint(self, printer, lhs):
        """Fortran print."""
 
        lhs_code = printer(lhs)
        if isinstance(self.arg, Array):
           init_value = printer(self.arg.arg)
        else:
           init_value = printer(self.arg)
        code_init = "{0} = shape({1})".format(lhs_code, init_value)
        
        return code_init


class Int(Function):
    """Represents a call to  numpy.int for code generation.

    arg : Variable,Float,Integer
    """
    def __new__(cls, arg):
        if not isinstance(arg,(Variable,NativeInteger, NativeFloat, NativeDouble, NativeComplex)):
            raise TypeError("Uknown type of  %s." % type(arg))
        return Basic.__new__(cls, arg)

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
    
    def fprint(self, printer, lhs):
        """Fortran print."""
        lhs_code = printer(lhs)
        init_value = printer(self.arg)
        code = "{0} = Int({1})".format(lhs_code, init_value)
        return code



class Zeros(Function):
    """Represents a call to numpy.zeros for code generation.

    shape : int, list, tuple
        int or list of integers

    dtype: str, DataType
        datatype for the constructed array

    Examples

    """
    # TODO improve
    def __new__(cls, shape, dtype=None):

        if isinstance(shape, list):
            # this is a correction. otherwise it is not working on LRZ
            if isinstance(shape[0], list):
                shape = Tuple(*(sympify(i) for i in shape[0]))
            else:
                shape = Tuple(*(sympify(i) for i in shape))
        elif isinstance(shape, int):
            shape = Tuple(sympify(shape))
        else:
            shape = shape

        if dtype is None:
            dtype = String('double')

        if isinstance(dtype, String):
            dtype = datatype('ndarray'+dtype.arg.replace('\'', ''))
        elif not isinstance(dtype, DataType):
            raise TypeError("datatype must be an instance of DataType.")

        return Basic.__new__(cls, shape, dtype)

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
    def init_value(self):
        dtype = self.dtype
        if isinstance(dtype, NativeInteger):
            value = 0
        elif isinstance(dtype, NativeFloat):
            value = 0.0
        elif isinstance(dtype, NativeDouble):
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
        if isinstance(self.shape, Tuple):
            # this is a correction. problem on LRZ
            shape_code = ', '.join('0:' + printer(i-1) for i in self.shape)
        else:
            shape_code = '0:' + printer(self.shape-1)

        init_value = printer(self.init_value)

        lhs_code = printer(lhs)

        code_alloc = "allocate({0}({1}))".format(lhs_code, shape_code)
        code_init = "{0} = {1}".format(lhs_code, init_value)
        code = "{0}\n{1}".format(code_alloc, code_init)
        return code


class Ones(Zeros):
    """Represents a call to numpy.ones for code generation.

    shape : int or list of integers

    Examples

    """
    @property
    def init_value(self):
        dtype = self.dtype
        if isinstance(dtype, NativeInteger):
            value = 1
        elif isinstance(dtype, NativeFloat):
            value = 1.0
        elif isinstance(dtype, NativeDouble):
            value = 1.0
        elif isinstance(dtype, NativeComplex):
            value = 1.0
        elif isinstance(dtype, NativeBool):
            value = BooleanTrue()
        else:
            raise TypeError('Unknown type')
        return value

