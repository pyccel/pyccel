# coding: utf-8

#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
Classes and methods that handle supported datatypes in C/Fortran.
"""

import numpy

from pyccel.utilities.metaclasses import Singleton

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
    'NativeRange',
    'NativeFloat',
    'NativeString',
    'NativeSymbol',
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
    'Float',
    'String',
    'Void',
	'NativeNumeric',
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
        16 : 'C_INT128_T'}, #no supported yet
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
iso_c_binding_shortcut_mapping = {
    'C_INT8_T'              : 'i8',
    'C_INT16_T'             : 'i16',
    'C_INT32_T'             : 'i32',
    'C_INT64_T'             : 'i64',
    'C_INT128_T'            : 'i128',
    'C_FLOAT'               : 'f32',
    'C_DOUBLE'              : 'f64',
    'C_LONG_DOUBLE'         : 'f128',
    'C_FLOAT_COMPLEX'       : 'c32',
    'C_DOUBLE_COMPLEX'      : 'c64',
    'C_LONG_DOUBLE_COMPLEX' : 'c128',
    'C_BOOL'                : 'b4'
}
default_precision = {'float': 8,
                    'int': numpy.dtype(int).alignment,
                    'integer': numpy.dtype(int).alignment,
                    'complex': 8,
                    'bool':4}
dtype_and_precision_registry = { 'float':('float', -1),
                                 'double':('float', -1),
                                 'real':('float', -1),
                                 'pythonfloat':('float', -1), # built-in float
                                 'float32':('float',4),
                                 'float64':('float',8),
                                 'pythoncomplex':('complex', -1),
                                 'complex':('complex', -1),  # to create numpy array with dtype='complex'
                                 'complex64':('complex',4),
                                 'complex128':('complex',8),
                                 'int8' :('int',1),
                                 'int16':('int',2),
                                 'int32':('int',4),
                                 'int64':('int',8),
                                 'int'  :('int', -1),
                                 'pythonint'  :('int', -1),
                                 'integer':('int',-1),
                                 'bool' :('bool',-1),
                                 'pythonbool' :('bool',-1)}


class DataType(metaclass=Singleton):
    """Base class representing native datatypes"""
    __slots__ = ()
    _name = '__UNDEFINED__'

    @property
    def name(self):
        return self._name

    def __str__(self):
        return str(self.name).lower()

    def __repr__(self):
        return str(self.__class__.__name__)+'()'

class NativeBool(DataType):
    """Class representing boolean datatype"""
    __slots__ = ()
    _name = 'Bool'

class NativeInteger(DataType):
    """Class representing integer datatype"""
    __slots__ = ()
    _name = 'Int'

class NativeFloat(DataType):
    """Class representing float datatype"""
    __slots__ = ()
    _name = 'Float'

class NativeComplex(DataType):
    """Class representing complex datatype"""
    __slots__ = ()
    _name = 'Complex'

NativeNumeric = (NativeBool(), NativeInteger(), NativeFloat(), NativeComplex())

class NativeString(DataType):
    """Class representing string datatype"""
    __slots__ = ()
    _name = 'String'

class NativeVoid(DataType):
    __slots__ = ()
    _name = 'Void'

class NativeNil(DataType):
    __slots__ = ()
    _name = 'Nil'

class NativeTuple(DataType):
    """Base class representing native datatypes"""
    __slots__ = ()
    _name = 'Tuple'

class NativeRange(DataType):
    __slots__ = ()
    _name = 'Range'

class NativeSymbol(DataType):
    __slots__ = ()
    _name = 'Symbol'


# TODO to be removed
class CustomDataType(DataType):
    __slots__ = ('_name',)

    def __init__(self, name='__UNDEFINED__'):
        self._name = name

class NativeGeneric(DataType):
    _name = 'Generic'


# ...
class VariableType(DataType):
    __slots__ = ('_alias','_rhs','_name')

    def __init__(self, rhs, alias):
        self._alias = alias
        self._rhs = rhs
        self._name = rhs._name

    @property
    def alias(self):
        return self._alias

class FunctionType(DataType):
    __slots__ = ('_domain','_codomain','_domains','_name')

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
Float          = NativeFloat()
Cmplx          = NativeComplex()
Void           = NativeVoid()
Nil            = NativeNil()
String         = NativeString()
_Symbol        = NativeSymbol()
Generic        = NativeGeneric()

dtype_registry = {'bool': Bool,
                  'int': Int,
                  'integer': Int,
                  'float'   : Float,
                  'complex': Cmplx,
                  'void': Void,
                  'nil': Nil,
                  'symbol': _Symbol,
                  '*': Generic,
                  'str': String}


class UnionType:
    """ Class representing multiple different possible
    datatypes for a function argument. If multiple
    arguments have union types then the result is a
    cross product of types
    """
    __slots__ = ('_args',)

    def __init__(self, args):
        self._args = args
        super().__init__()

    @property
    def args(self):
        return self._args


def DataTypeFactory(name, argnames=["_name"],
                    BaseClass=CustomDataType,
                    prefix=None,
                    alias=None,
                    is_iterable=False,
                    is_with_construct=False,
                    is_polymorphic=False):
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
    elif isinstance(dtype, NativeRange):
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

# TODO check the use of Floats
def datatype(arg):
    """Returns the datatype singleton for the given dtype.

    arg : str or pyccel expression
        If a str ('bool', 'int', 'float','complex', or 'void'), return the
        singleton for the corresponding dtype. If a pyccel expression, return
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
    This function takes a datatype and returns a pyccel datatype as a string

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
        elif dtype== 'float':
            return 'float'
        else:
            return dtype
    if isinstance(dtype, NativeInteger):
        return 'integer'
    elif isinstance(dtype, NativeFloat):
        return 'float'
    elif isinstance(dtype, NativeComplex):
        return 'complex'
    elif isinstance(dtype, NativeBool):
        return 'bool'
    else:
        raise TypeError('Unknown datatype {0}'.format(str(dtype)))
