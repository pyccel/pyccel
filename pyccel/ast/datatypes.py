# coding: utf-8

#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
Classes and methods that handle supported datatypes in C/Fortran.
"""
from functools import lru_cache

import numpy

from pyccel.utilities.metaclasses import ArgumentSingleton, Singleton

# TODO [YG, 12.03.2020] verify why we need all these types
# NOTE: symbols not used in pyccel are commented out
__all__ = (
#
# --------- CLASSES -----------
#
    'CustomDataType',
    'DataType',
    'NativeBool',
    'NativeComplex',
    'NativeFloat',
    'NativeGeneric',
    'NativeInteger',
    'NativeTuple',
    'NativeNil',
    'NativeRange',
    'NativeString',
    'NativeSymbol',
    'NativeVoid',
    'UnionType',
    'DataTypeFactory',
#
# --------- FUNCTIONS -----------
#
    'datatype',
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
        -1 : "C_BOOL"}
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
    'C_BOOL'                : 'b1'
}

#==============================================================================

class DataType(metaclass=ArgumentSingleton):
    """
    Base class representing native datatypes.

    The base class from which all data types must inherit.
    """
    __slots__ = ()
    _name = '__UNDEFINED__'

    @property
    def name(self):
        """
        Get the name of the datatype.

        Get the name of the datatype.
        """
        return self._name

    def __str__(self):
        return str(self.name).lower()

    def __repr__(self):
        return str(self.__class__.__name__)+'()'

    def __reduce__(self):
        """
        Function called during pickling.

        For more details see : https://docs.python.org/3/library/pickle.html#object.__reduce__.
        This function is necessary to ensure that DataTypes remain singletons.

        Returns
        -------
        callable
            A callable to create the object.
        args
            A tuple containing any arguments to be passed to the callable.
        """
        return (self.__class__, ())

class NativeBool(DataType, metaclass=Singleton):
    """Class representing boolean datatype"""
    __slots__ = ()
    _name = 'Bool'

    @lru_cache
    def __add__(self, other):
        if other in NativeNumeric:
            return other
        else:
            return NotImplemented

class NativeInteger(DataType, metaclass=Singleton):
    """Class representing integer datatype"""
    __slots__ = ()
    _name = 'Int'

    @lru_cache
    def __add__(self, other):
        if other in NativeNumeric:
            if other is NativeBool():
                return self
            else:
                return other
        else:
            return NotImplemented

class NativeFloat(DataType, metaclass=Singleton):
    """Class representing float datatype"""
    __slots__ = ()
    _name = 'Float'

    @lru_cache
    def __add__(self, other):
        if other in NativeNumeric:
            if other is NativeComplex():
                return other
            else:
                return self
        else:
            return NotImplemented

class NativeComplex(DataType, metaclass=Singleton):
    """Class representing complex datatype"""
    __slots__ = ()
    _name = 'Complex'

    @lru_cache
    def __add__(self, other):
        if other in NativeNumeric:
            return self
        else:
            return NotImplemented

NativeNumeric = (NativeBool(), NativeInteger(), NativeFloat(), NativeComplex())
NativeNumericTypes = (NativeBool, NativeInteger, NativeFloat, NativeComplex)

class NativeString(DataType, metaclass=Singleton):
    """Class representing string datatype"""
    __slots__ = ()
    _name = 'String'

class NativeVoid(DataType, metaclass=Singleton):
    __slots__ = ()
    _name = 'Void'

class NativeNil(DataType, metaclass=Singleton):
    __slots__ = ()
    _name = 'Nil'

class NativeTuple(DataType):
    """Base class representing native datatypes"""
    __slots__ = ()
    _name = 'Tuple'

class NativeHomogeneousTuple(NativeTuple, metaclass = ArgumentSingleton):
    __slots__ = ()

    def __init__(self):
        super().__init__()

class NativeInhomogeneousTuple(NativeTuple, metaclass = ArgumentSingleton):
    __slots__ = ('_dtypes',)

    def __init__(self, *dtypes):
        self._dtypes = dtypes
        super().__init__()

    @property
    def name(self):
        datatypes = ', '.join(d.name for d in self._dtypes)
        return f'tuple[{datatypes}]'

    def __getitem__(self, i):
        return self._dtypes[i]

class NativeHomogeneousList(DataType):
	__slots__ = ()
	_name = 'List'

    def __init__(self):
		super().__init__()


class NativeRange(DataType):
    __slots__ = ()
    _name = 'Range'

class NativeSymbol(DataType, metaclass=Singleton):
    __slots__ = ()
    _name = 'Symbol'

class CustomDataType(DataType, metaclass=Singleton):
    """
    Class from which user-defined types inherit.

    A general class for custom data types which is used as a
    base class when a user defines their own type using classes.
    """
    __slots__ = ()

class NativeGeneric(DataType):
    __slots__ = ()
    _name = 'Generic'

    @lru_cache
    def __add__(self, other):
        return other

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

dtype_and_precision_registry = { 'float' : (Float, -1),
                                 'double' : (Float, -1),
                                 'real' : (Float, -1),
                                 'float32' : (Float,4),
                                 'float64' : (Float,8),
                                 'f4' : (Float,4),
                                 'f8' : (Float,8),
                                 'complex' : (Cmplx, -1),
                                 'complex64' : (Cmplx,4),
                                 'complex128' : (Cmplx,8),
                                 'c8' : (Cmplx,4),
                                 'c16' : (Cmplx,8),
                                 'int8' :(Int,1),
                                 'int16' : (Int,2),
                                 'int32' : (Int,4),
                                 'int64' : (Int,8),
                                 'i1' :(Int,1),
                                 'i2' : (Int,2),
                                 'i4' : (Int,4),
                                 'i8' : (Int,8),
                                 'int'  :(Int, -1),
                                 'integer' : (Int,-1),
                                 'bool' :(Bool,-1),
                                 'b1' :(Bool,-1),
                                 'void' : (Void, 0),
                                 'nil' : (Nil, 0),
                                 'symbol' : (_Symbol, 0),
                                 '*' : (Generic, 0),
                                 'str' : (String, 0),
                                 }

default_precision = {Float : 8,
                     Int : numpy.dtype(int).alignment,
                     Cmplx : 8,
                     Bool : -1}

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
                    prefix=None):
    """
    Create a new data class.

    Create a new data class which sub-classes a DataType. This provides
    a new data type which can be used, for example, for class types.

    Parameters
    ----------
    name : str
        The name of the new class.

    argnames : list of str
        A list of all the arguments for the new class.

    BaseClass : type inheriting from DataType
        The class from which the new type will be sub-classed.

    prefix : str
        A prefix which will be added to the class name.

    Returns
    -------
    type
        A new DataType class.
    """
    def __init__(self, **kwargs):
        for key, value in list(kwargs.items()):
            # here, the argnames variable is the one passed to the
            # DataTypeFactory call
            if key not in argnames:
                raise TypeError("Argument %s not valid for %s"
                    % (key, self.__class__.__name__))
            setattr(self, key, value)
        BaseClass.__init__(self)

    if prefix is None:
        prefix = 'Pyccel'
    else:
        prefix = 'Pyccel{0}'.format(prefix)

    newclass = type(prefix + name, (BaseClass,),
                    {"__init__": __init__,
                     "_name": name,
                     "prefix": prefix,
                     "alias": name})

    dtype_and_precision_registry[name] = (newclass(), 0)
    return newclass

def datatype(arg):
    """
    Get the datatype indicated by a string.

    Return the datatype singleton for the dtype described
    by the argument.

    Parameters
    ----------
    arg : str
        Return the singleton for the corresponding dtype.

    Returns
    -------
    DataType
        The data type described by the string.
    """
    if isinstance(arg, str):
        if arg not in dtype_and_precision_registry:
            raise ValueError("Unrecognized datatype " + arg)
        return dtype_and_precision_registry[arg][0]
    else:
        raise TypeError('Expecting a DataType')

def str_dtype(dtype):

    """
    Get a string describing a datatype.

    This function takes a pyccel datatype and returns a string which describes it.

    Parameters
    ----------
    dtype : DataType
        The datatype.

    Returns
    -------
    str
        A description of the data type.

    Examples
    --------
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
    elif isinstance(dtype, CustomDataType):
        return dtype.name
    else:
        raise TypeError('Unknown datatype {0}'.format(str(dtype)))
