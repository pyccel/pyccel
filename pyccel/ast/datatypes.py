# coding: utf-8


from .basic import Basic

from sympy.core.singleton import Singleton
from sympy.core.compatibility import with_metaclass
from sympy import sympify
from sympy import ImmutableDenseMatrix


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

class NativeFloat(DataType):
    _name = 'Float'
    pass

class NativeDouble(DataType):
    _name = 'Double'
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

class NativeFloatList(NativeFloat, NativeList):
    _name = 'FloatList'
    pass

class NativeDoubleList(NativeDouble, NativeList):
    _name = 'DoubleList'
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
    _name = 'NdArrayInt'
    pass

class NdArrayFloat(NdArray, NativeFloat):
    _name = 'NdArrayFloat'
    pass

class NdArrayDouble(NdArray, NativeDouble):
    _name = 'NdArrayDouble'
    pass

class NdArrayComplex(NdArray, NativeComplex):
    _name = 'NdArrayComplex'
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
Float   = NativeFloat()
Double  = NativeDouble()
Complex = NativeComplex()
Void    = NativeVoid()
Nil     = NativeNil()
String  = NativeString()
_Symbol = NativeSymbol()
IntegerList = NativeIntegerList()
FloatList = NativeFloatList()
DoubleList = NativeDoubleList()
ComplexList = NativeComplexList()
NdArray = NdArray()
NdArrayInt = NdArrayInt()
NdArrayDouble = NdArrayDouble()
NdArrayFloat = NdArrayFloat()
NdArrayComplex = NdArrayComplex()
Generic    = NativeGeneric()


dtype_registry = {'bool': Bool,
                  'int': Int,
                  'float': Float,
                  'double': Double,
                  'complex': Complex,
                  'void': Void,
                  'nil': Nil,
                  'symbol': _Symbol,
                  '*int': IntegerList,
                  '*float': FloatList,
                  '*double': DoubleList,
                  '*complex': ComplexList,
                  'ndarrayint': NdArrayInt,
                  'ndarrayfloat': NdArrayFloat,
                  'ndarraydouble': NdArrayDouble,
                  'ndarraycomplex': NdArrayComplex,
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

# TODO check the use of floats
def datatype(arg):
    """Returns the datatype singleton for the given dtype.

    arg : str or sympy expression
        If a str ('bool', 'int', 'float', 'double', or 'void'), return the
        singleton for the corresponding dtype. If a sympy expression, return
        the datatype that best fits the expression. This is determined from the
        assumption system. For more control, use the `DataType` class directly.

    Returns:
        DataType

    """
    def infer_dtype(arg):
        if arg.is_integer:
            return Int
        elif arg.is_Boolean:
            return Bool
        else:
            return Double

    if isinstance(arg, str):
        if arg.lower() not in dtype_registry:
            raise ValueError("Unrecognized datatype " + arg)
        return dtype_registry[arg]
    elif isinstance(arg, (Variable, IndexedVariable, IndexedElement)):
        if isinstance(arg.dtype, DataType):
            return dtype_registry[arg.dtype.name.lower()]
        else:
            raise TypeError('Expecting a DataType')
    else:
        arg = sympify(arg)
        if isinstance(arg, ImmutableDenseMatrix):
            dts = [infer_dtype(i) for i in arg]
            if all([i is Bool for i in dts]):
                return Bool
            elif all([i is Int for i in dts]):
                return Int
            else:
                return Double
        else:
            return infer_dtype(arg)

from .core import Variable, IndexedVariable, IndexedElement
