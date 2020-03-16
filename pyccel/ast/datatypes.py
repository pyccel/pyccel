# coding: utf-8


from .basic import Basic

from sympy.core.singleton import Singleton
from sympy.core.compatibility import with_metaclass
from sympy import Eq, Ne, Lt, Gt, Le, Ge

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
    'NativeComplexList',
    'NativeGeneric',
    'NativeInteger',
    'NativeIntegerList',
    'NativeList',
#    'NativeNil',
#    'NativeParallelRange',
    'NativeRange',
    'NativeReal',
    'NativeRealList',
    'NativeString',
    'NativeSymbol',
    'NativeTensor',
    'NativeVoid',
#    'NdArray',
#    'NdArrayBool',
#    'NdArrayComplex',
#    'NdArrayInt',
#    'NdArrayReal',
    'UnionType',
    'VariableType',
    'DataTypeFactory',
#
# --------- FUNCTIONS -----------
#
    'datatype',
#    'get_default_value',
    'is_iterable_datatype',
    'is_pyccel_datatype',
    'is_with_construct_datatype',
    'sp_dtype',
    'str_dtype',
#
# --------- VARIABLES -----------
#
    'Bool',
    'Complex',
    'ComplexList',
    'Generic',
    'Int',
    'IntegerList',
#    'NdArray',
#    'NdArrayBool',
#    'NdArrayComplex',
#    'NdArrayInt',
#    'NdArrayReal',
    'Nil',
    'Real',
    'RealList',
    'String',
    'Void',
#    '_Symbol',
    'default_precision',
    'dtype_and_precision_registry',
    'dtype_registry'
)

#==============================================================================
default_precision = {'real': 8, 'int': 8, 'complex': 8, 'bool':4, 'float':8}
dtype_and_precision_registry = { 'real':('real',8),
                                 'double':('real',8),
                                 'float':('real',8),       # sympy.Float
                                 'pythonfloat':('real',8), # built-in float
                                 'float32':('real',4),
                                 'float64':('real',8),
                                 'complex':('complex',8),
                                 'complex64':('complex',4),
                                 'complex128':('complex',8),
                                 'int8' :('int',1),
                                 'int16':('int',2),
                                 'int32':('int',4),
                                 'int64':('int',8),
                                 'int'  :('int',8),
                                 'integer':('int',4),
                                 'bool' :('bool',4)}


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
    pass

class NativeInteger(DataType):
    _name = 'Int'
    pass

class NativeReal(DataType):
    _name = 'Real'
    pass

class NativeComplex(DataType):
    _name = 'Complex'
    pass

class NativeString(DataType):
    _name = 'String'
    pass

class NativeVoid(DataType):
    _name = 'Void'
    pass

class NativeNil(DataType):
    _name = 'Nil'
    pass

class NativeList(DataType):
    _name = 'List'
    pass

class NativeIntegerList(NativeInteger, NativeList):
    _name = 'IntegerList'
    pass

class NativeRealList(NativeReal, NativeList):
    _name = 'RealList'
    pass

class NativeComplexList(NativeComplex, NativeList):
    _name = 'ComplexList'
    pass

class NativeRange(DataType):
    _name = 'Range'
    pass

class NativeTensor(DataType):
    _name = 'Tensor'
    pass

class NativeParallelRange(NativeRange):
    _name = 'ParallelRange'
    pass

class NativeSymbol(DataType):
    _name = 'Symbol'
    pass

class NdArray(DataType):
    _name = 'NdArray'
    pass

class NdArrayInt(NdArray, NativeInteger):
    _name = 'int'
    pass

class NdArrayReal(NdArray, NativeReal):
    _name = 'real'
    pass


class NdArrayComplex(NdArray, NativeComplex):
    _name = 'complex'
    pass

class NdArrayBool(NdArray, NativeBool):
    _name = 'bool'
    pass

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



Bool    = NativeBool()
Int     = NativeInteger()
Real    = NativeReal()
Complex = NativeComplex()
Void    = NativeVoid()
Nil     = NativeNil()
String  = NativeString()
_Symbol = NativeSymbol()
IntegerList = NativeIntegerList()
RealList = NativeRealList()
ComplexList = NativeComplexList()
NdArray = NdArray()
NdArrayInt = NdArrayInt()
NdArrayReal = NdArrayReal()
NdArrayComplex = NdArrayComplex()
NdArrayBool = NdArrayBool()
Generic    = NativeGeneric()


dtype_registry = {'bool': Bool,
                  'int': Int,
                  'integer': Int,
                  'real'   : Real,
                  'complex': Complex,
                  'void': Void,
                  'nil': Nil,
                  'symbol': _Symbol,
                  '*int': IntegerList,
                  '*real': RealList,
                  '*complex': ComplexList,
                  'ndarrayint': NdArrayInt,
                  'ndarrayinteger':NdArrayInt,
                  'ndarrayreal': NdArrayReal,
                  'ndarraycomplex': NdArrayComplex,
                  'ndarraybool': NdArrayBool,
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
#    if not isinstance(expr, DataType):
#        raise TypeError('Expecting a DataType instance')
#    name = expr.__class__.__name__
#    return name.startswith('Pyccel')

# TODO improve and remove try/except
def is_iterable_datatype(dtype):
    """Returns True if dtype is an iterable class."""
    try:
        if is_pyccel_datatype(dtype):
            return dtype.is_iterable
        elif isinstance(dtype, (NativeRange, NativeTensor)):
            return True
        else:
            return False
    except:
        return False


def get_default_value(dtype):
    """Returns the default value of a native datatype."""
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


# TODO improve and remove try/except
def is_with_construct_datatype(dtype):
    """Returns True if dtype is an with_construct class."""
    try:
        if is_pyccel_datatype(dtype):
            return dtype.is_with_construct
        else:
            return False
    except:
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
        return dtype_registry[arg.dtype.name.lower()]
    else:
        raise TypeError('Expecting a DataType')


def sp_dtype(expr):
    """
    return the datatype of a sympy types expression

    """
    if expr.is_integer:
        return 'int'
    elif expr.is_real:
        return 'real'
    elif expr.is_complex:
        return 'complex'
    elif expr.is_Boolean:
        return 'bool'
    elif isinstance(expr,(Eq, Ne, Lt, Gt, Le, Ge)):
        return 'bool'
    else:
        raise TypeError('Unknown datatype {0}'.format(str(expr)))


def str_dtype(dtype):

    """
    return a sympy datatype as string
    dtype: str, Native Type

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


