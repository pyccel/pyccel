# coding: utf-8

#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
Classes and methods that handle supported datatypes in C/Fortran.
"""

from .basic import Basic

from sympy.core.singleton import Singleton
from sympy.core.compatibility import with_metaclass

import numpy

# TODO [YG, 12.03.2020] verify why we need all these types
# NOTE: symbols not used in pyccel are commented out
__all__ = (
#
# --------- CLASSES -----------
#
    'CustomDataType',
    'DataType',
    'FunctionType',
    'NativeBool',
    'NativeComplex',
    'NativeGeneric',
    'NativeInteger',
    'NativeTuple',
#    'NativeNil',
#    'NativeParallelRange',
    'NativeRange',
    'NativeReal',
    'NativeString',
    'NativeSymbol',
    'NativeTensor',
    'NativeVoid',
    'UnionType',
    'VariableType',
    'DataTypeFactory',
#
# --------- FUNCTIONS -----------
#
    'datatype',
    'is_iterable_datatype',
    'is_pyccel_datatype',
    'is_with_construct_datatype',
    'str_dtype',
#
# --------- VARIABLES -----------
#
    'Bool',
    'Cmplx',
    'Generic',
    'Int',
    'Nil',
    'Real',
    'String',
    'Void',
#    '_Symbol',
    'default_precision',
    'dtype_and_precision_registry',
    'dtype_registry'
)

#==============================================================================
iso_c_binding = {
    "integer" : {
        1  : 'C_INT8_T',
        2  : 'C_INT16_T',
        4  : 'C_INT32_T',
        8  : 'C_INT64_T',
        16 : 'C_INT128'}, #no supported yet
    "real"    : {
        4  : 'C_FLOAT',
        8  : 'C_DOUBLE',
        16 : 'C_LONG_DOUBLE'},
    "complex" : {
        4  : 'C_FLOAT_COMPLEX',
        8  : 'C_DOUBLE_COMPLEX',
        16 : 'C_LONG_DOUBLE_COMPLEX'},
    "logical" : {
        4  : "C_BOOL"}
}

default_precision = {'real': 8,
                    'int': numpy.dtype(int).alignment,
                    'integer': numpy.dtype(int).alignment,
                    'complex': 8,
                    'bool':4,
                    'float':8}
dtype_and_precision_registry = { 'real':('real',default_precision['float']),
                                 'double':('real',default_precision['float']),
                                 'float':('real',default_precision['float']),       # sympy.Float
                                 'pythonfloat':('real',default_precision['float']), # built-in float
                                 'float32':('real',4),
                                 'float64':('real',8),
                                 'pythoncomplex':('complex',default_precision['complex']),
                                 'complex':('complex',default_precision['complex']),  # to create numpy array with dtype='complex'
                                 'complex64':('complex',4),
                                 'complex128':('complex',8),
                                 'int8' :('int',1),
                                 'int16':('int',2),
                                 'int32':('int',4),
                                 'int64':('int',8),
                                 'int'  :('int', default_precision['int']),
                                 'pythonint'  :('int', default_precision['int']),
                                 'integer':('int',default_precision['int']),
                                 'bool' :('bool',default_precision['bool']),
                                 'pythonbool' :('bool',default_precision['bool'])}


class DataType(with_metaclass(Singleton, Basic)):
    """Base class representing native datatypes"""
    _name = '__UNDEFINED__'

    @property
    def name(self):
        return self._name

    def __str__(self):
        return str(self.name).lower()

class NativeBool(DataType):
    _name = 'Bool'

class NativeInteger(DataType):
    _name = 'Int'

class NativeReal(DataType):
    _name = 'Real'

class NativeComplex(DataType):
    _name = 'Complex'

class NativeString(DataType):
    _name = 'String'

class NativeVoid(DataType):
    _name = 'Void'

class NativeNil(DataType):
    _name = 'Nil'

class NativeTuple(DataType):
    """Base class representing native datatypes"""
    _name = 'Tuple'

class NativeRange(DataType):
    _name = 'Range'

class NativeTensor(DataType):
    _name = 'Tensor'

class NativeParallelRange(NativeRange):
    _name = 'ParallelRange'

class NativeSymbol(DataType):
    _name = 'Symbol'


# TODO to be removed
class CustomDataType(DataType):
    _name = '__UNDEFINED__'

    def __init__(self, name='__UNDEFINED__'):
        self._name = name

class NativeGeneric(DataType):
    _name = 'Generic'
    pass


# ...
class VariableType(DataType):

    def __init__(self, rhs, alias):
        self._alias = alias
        self._rhs = rhs
        self._name = rhs._name

    @property
    def alias(self):
        return self._alias

class FunctionType(DataType):

    def __init__(self, domains):
        self._domain = domains[0]
        self._codomain = domains[1:]
        self._domains = domains
        self._name = ' -> '.join('{}'.format(V) for V in self._domains)

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain
# ...



Bool           = NativeBool()
Int            = NativeInteger()
Real           = NativeReal()
Cmplx          = NativeComplex()
Void           = NativeVoid()
Nil            = NativeNil()
String         = NativeString()
_Symbol        = NativeSymbol()
Generic        = NativeGeneric()

dtype_registry = {'bool': Bool,
                  'int': Int,
                  'integer': Int,
                  'real'   : Real,
                  'complex': Cmplx,
                  'void': Void,
                  'nil': Nil,
                  'symbol': _Symbol,
                  '*': Generic,
                  'str': String}


class UnionType(Basic):

    def __new__(cls, args):
        return Basic.__new__(cls, args)

    @property
    def args(self):
        return self._args[0]


def DataTypeFactory(name, argnames=["_name"],
                    BaseClass=CustomDataType,
                    prefix=None,
                    alias=None,
                    is_iterable=False,
                    is_with_construct=False,
                    is_polymorphic=True):
    def __init__(self, **kwargs):
        for key, value in list(kwargs.items()):
            # here, the argnames variable is the one passed to the
            # DataTypeFactory call
            if key not in argnames:
                raise TypeError("Argument %s not valid for %s"
                    % (key, self.__class__.__name__))
            setattr(self, key, value)
        BaseClass.__init__(self, name=name[:-len("Class")])

    if prefix is None:
        prefix = 'Pyccel'
    else:
        prefix = 'Pyccel{0}'.format(prefix)

    newclass = type(prefix + name, (BaseClass,),
                    {"__init__":          __init__,
                     "_name":             name,
                     "prefix":            prefix,
                     "alias":             alias,
                     "is_iterable":       is_iterable,
                     "is_with_construct": is_with_construct,
                     "is_polymorphic":    is_polymorphic})
    return newclass

def is_pyccel_datatype(expr):
    return isinstance(expr, CustomDataType)

def is_iterable_datatype(dtype):
    """Returns True if dtype is an iterable class."""
    if is_pyccel_datatype(dtype):
        return dtype.is_iterable
    elif isinstance(dtype, (NativeRange, NativeTensor)):
        return True
    else:
        return False


# TODO improve
def is_with_construct_datatype(dtype):
    """Returns True if dtype is an with_construct class."""
    if is_pyccel_datatype(dtype):
        return dtype.is_with_construct
    else:
        return False

# TODO check the use of Reals
def datatype(arg):
    """Returns the datatype singleton for the given dtype.

    arg : str or sympy expression
        If a str ('bool', 'int', 'real','complex', or 'void'), return the
        singleton for the corresponding dtype. If a sympy expression, return
        the datatype that best fits the expression. This is determined from the
        assumption system. For more control, use the `DataType` class directly.

    Returns:
        DataType

    """


    if isinstance(arg, str):
        if arg.lower() not in dtype_registry:
            raise ValueError("Unrecognized datatype " + arg)
        return dtype_registry[arg]
    if isinstance(arg, DataType):
        return dtype_registry[arg.name.lower()]
    else:
        raise TypeError('Expecting a DataType')

def str_dtype(dtype):

    """
    This function takes a datatype and returns a sympy datatype as a string

    Example
    -------
    >>> str_dtype('int')
    'integer'
    >>> str_dtype(NativeInteger())
    'integer'

    """
    if isinstance(dtype, str):
        if dtype == 'int':
            return 'integer'
        elif dtype== 'real':
            return 'real'
        else:
            return dtype
    if isinstance(dtype, NativeInteger):
        return 'integer'
    elif isinstance(dtype, NativeReal):
        return 'real'
    elif isinstance(dtype, NativeComplex):
        return 'complex'
    elif isinstance(dtype, NativeBool):
        return 'bool'
    else:
        raise TypeError('Unknown datatype {0}'.format(str(dtype)))
