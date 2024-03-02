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

from pyccel.utilities.metaclasses import Singleton, ArgumentSingleton

__all__ = (
        # ------------ Super classes ------------
        'PyccelType',
        'PrimitiveType',
        'FixedSizeType',
        'ContainerType',
        # ------------ Primitive types ------------
        'PyccelBooleanType',
        'PyccelIntegerType',
        'PyccelFloatingPointType',
        'PyccelComplexType',
        'PyccelCharacterType',
        # ------------ Fixed size types ------------
        'FixedSizeNumericType',
        'PythonNativeNumericTypes',
        'PythonNativeBool',
        'PythonNativeInt',
        'PythonNativeFloat',
        'PythonNativeComplex',
        'VoidType',
        'GenericType',
        'SymbolicType',
        'CharType',
        # ------------ Container types ------------
        'TupleType',
        'HomogeneousContainerType',
        'StringType',
        'HomogeneousTupleType',
        'HomogeneousListType',
        'CustomDataType',
        'InhomogeneousTupleType',
        'DictType',
        # ---------- Functions -------------------
        'DataTypeFactory',
)

#==============================================================================
class PrimitiveType(metaclass=Singleton):
    """
    Base class representing types of datatypes.

    The base class representing the category of datatype to which a FixedSizeType
    may belong (e.g. integer, floating point).
    """
    __slots__ = ()
    _name = '__UNDEFINED__'

    def __str__(self):
        return self._name

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

class PyccelBooleanType(PrimitiveType):
    """
    Class representing a boolean datatype.

    Class representing a boolean datatype.
    """
    __slots__ = ()
    _name = 'boolean'

class PyccelIntegerType(PrimitiveType):
    """
    Class representing an integer datatype.

    Class representing an integer datatype.
    """
    __slots__ = ()
    _name = 'integer'

class PyccelFloatingPointType(PrimitiveType):
    """
    Class representing a floating point datatype.

    Class representing a floating point datatype.
    """
    __slots__ = ()
    _name = 'floating point'

class PyccelComplexType(PrimitiveType):
    """
    Class representing a complex datatype.

    Class representing a complex datatype.
    """
    __slots__ = ()
    _name = 'complex'

class PyccelCharacterType(PrimitiveType):
    """
    Class representing a character datatype.

    Class representing a character datatype.
    """
    __slots__ = ()
    _name = 'character'

#==============================================================================

class PyccelType:
    """
    Base class representing the type of an object.

    Base class representing the type of an object from which all
    types must inherit. A type must contain enough information to
    describe the declaration type in a low-level language.

    Types contain an addition operator. The operator indicates the type that
    is expected when calling an arithmetic operator on objects of these types.

    Where applicable, types also contain an and operator. The operator indicates the type that
    is expected when calling a bitwise comparison operator on objects of these types.
    """
    __slots__ = ()

    def __str__(self):
        return self._name #pylint: disable=no-member

    def switch_basic_type(self, new_type):
        """
        Change the basic type to the new type.

        Change the basic type to the new type. In the case of a FixedSizeType the
        switch will replace the type completely, directly returning the new type.
        In the case of a homogeneous container type, a new container type will be
        returned whose underlying elements are of the new type. This method is not
        implemented for inhomogeneous containers.

        Parameters
        ----------
        new_type : PyccelType
            The new basic type.

        Returns
        -------
        PyccelType
            The new type.
        """
        raise NotImplementedError(f"switch_basic_type not implemented for {type(self)}")

#==============================================================================

class FixedSizeType(PyccelType, metaclass=Singleton):
    """
    Base class representing a built-in scalar datatype.

    The base class representing a built-in scalar datatype which can be
    represented in memory. E.g. int32, int64.
    """
    __slots__ = ()

    @property
    def datatype(self):
        """
        The datatype of the object.

        The datatype of the object.
        """
        return self

    @property
    def primitive_type(self):
        """
        The datatype category of the object.

        The datatype category of the object (e.g. integer, floating point).
        """
        return self._primitive_type # pylint: disable=no-member

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

    def switch_basic_type(self, new_type):
        """
        Change the basic type to the new type.

        Change the basic type to the new type. In the case of a FixedSizeType the
        switch will replace the type completely, directly returning the new type.

        Parameters
        ----------
        new_type : PyccelType
            The new basic type.

        Returns
        -------
        PyccelType
            The new type.
        """
        assert isinstance(new_type, FixedSizeType)
        return new_type

class FixedSizeNumericType(FixedSizeType):
    """
    Base class representing a scalar numeric datatype.

    The base class representing a scalar numeric datatype which can be
    represented in memory. E.g. int32, int64.
    """
    __slots__ = ()

    @property
    def precision(self):
        """
        Precision of the datatype of the object.

        The precision of the datatype of the object. This number is related to the
        number of bytes that the datatype takes up in memory (e.g. `float64` has
        precision = 8 as it takes up 8 bytes, `complex128` has precision = 8 as
        it is comprised of two `float64` objects. The precision is equivalent to
        the `kind` parameter in Fortran.
        """
        return self._precision # pylint: disable=no-member

class PythonNativeNumericTypes(FixedSizeNumericType):
    """
    Base class representing a built-in scalar numeric datatype.

    Base class representing a built-in scalar numeric datatype.
    """
    __slots__ = ()

class PythonNativeBool(PythonNativeNumericTypes):
    """
    Class representing Python's native boolean type.

    Class representing Python's native boolean type.
    """
    __slots__ = ()
    _name = 'bool'
    _primitive_type = PyccelBooleanType()
    _precision = -1

    @lru_cache
    def __add__(self, other):
        if isinstance(other, PythonNativeBool):
            return PythonNativeInt()
        elif isinstance(other, PythonNativeNumericTypes):
            return other
        else:
            return NotImplemented

    @lru_cache
    def __and__(self, other):
        if isinstance(other, PythonNativeBool):
            return PythonNativeBool()
        elif isinstance(other, PythonNativeNumericTypes):
            return other
        else:
            return NotImplemented

class PythonNativeInt(PythonNativeNumericTypes):
    """
    Class representing Python's native integer type.

    Class representing Python's native integer type.
    """
    __slots__ = ()
    _name = 'int'
    _primitive_type = PyccelIntegerType()
    _precision = numpy.dtype(int).alignment

    @lru_cache
    def __add__(self, other):
        if isinstance(other, PythonNativeBool):
            return self
        elif isinstance(other, PythonNativeNumericTypes):
            return other
        else:
            return NotImplemented

    @lru_cache
    def __and__(self, other):
        if isinstance(other, PythonNativeNumericTypes):
            return self
        else:
            return NotImplemented

class PythonNativeFloat(PythonNativeNumericTypes):
    """
    Class representing Python's native floating point type.

    Class representing Python's native floating point type.
    """
    __slots__ = ()
    _name = 'float'
    _primitive_type = PyccelFloatingPointType()
    _precision = 8

    @lru_cache
    def __add__(self, other):
        if isinstance(other, PythonNativeComplex):
            return other
        elif isinstance(other, PythonNativeNumericTypes):
            return self
        else:
            return NotImplemented

class PythonNativeComplex(PythonNativeNumericTypes):
    """
    Class representing Python's native complex type.

    Class representing Python's native complex type.
    """
    __slots__ = ('_element_type',)
    _name = 'complex'
    _primitive_type = PyccelComplexType()
    _precision = 8

    @lru_cache
    def __add__(self, other):
        if isinstance(other, PythonNativeNumericTypes):
            return self
        else:
            return NotImplemented

    @property
    def element_type(self):
        """
        The type of an element of the complex.

        The type of an element of the complex. In other words, the type
        of the floats which comprise the complex type.
        """
        return PythonNativeFloat()

class VoidType(FixedSizeType):
    """
    Class representing a void datatype.

    Class representing a void datatype. This class is especially useful
    in the C-Python wrapper when a `void*` type is needed to collect
    pointers from Fortran.
    """
    __slots__ = ()
    _name = 'void'
    _primitive_type = None

class GenericType(FixedSizeType):
    """
    Class representing a generic datatype.

    Class representing a generic datatype. This datatype is
    useful for describing the type of an empty container (list/tuple/etc)
    or an argument which can accept any type (e.g. MPI arguments).
    """
    __slots__ = ()
    _name = 'Generic'
    _primitive_type = None

    @lru_cache
    def __add__(self, other):
        return other

class SymbolicType(FixedSizeType):
    """
    Class representing the datatype of a placeholder symbol.

    Class representing the datatype of a placeholder symbol. This type should
    be used for objects which will not appear in the generated code but are
    used to identify objects (e.g. Type aliases).
    """
    __slots__ = ()
    _name = 'Symbolic'
    _primitive_type = None

class CharType(FixedSizeType):
    """
    Class representing a char type in C/Fortran.

    Class representing a char type in C/Fortran. This datatype is
    useful for describing strings.
    """
    __slots__ = ()
    _name = 'char'
    _primitive_type = PyccelCharacterType

#==============================================================================

class ContainerType(PyccelType):
    """
    Base class representing a type which contains objects of other types.

    Base class representing a type which contains objects of other types.
    E.g. classes, arrays, etc.
    """
    __slots__ = ()

#==============================================================================

class TupleType:
    """
    Base class representing tuple datatypes.

    The class from which tuple datatypes must inherit.
    """
    __slots__ = ()
    _name = 'tuple'

#==============================================================================

class HomogeneousContainerType(ContainerType):
    """
    Base class representing a datatype which contains multiple elements of a given type.

    Base class representing a datatype which contains multiple elements of a given type.
    This is the case for objects such as arrays, lists, etc.
    """
    __slots__ = ()

    @property
    def datatype(self):
        """
        The datatype of the object.

        The datatype of the object.
        """
        return self.element_type.datatype

    @property
    def primitive_type(self):
        """
        The datatype category of elements of the object.

        The datatype category of elements of the object (e.g. integer, floating point).
        """
        return self.element_type.primitive_type

    @property
    def precision(self):
        """
        Precision of the elements of the object.

        The precision of the elements of the object. This number is related to the
        number of bytes that the datatype takes up in memory (e.g. `float64` has
        precision = 8 as it takes up 8 bytes, `complex128` has precision = 8 as
        it is comprised of two `float64` objects. The precision is equivalent to
        the `kind` parameter in Fortran.
        """
        return self.element_type.precision

    @property
    def element_type(self):
        """
        The type of elements of the object.

        The PyccelType describing an element of the container.
        """
        return self._element_type # pylint: disable=no-member

    def __str__(self):
        return f'{self._name}[{self._element_type}]' # pylint: disable=no-member

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
        return (self.__class__, (self.element_type,))

    def switch_basic_type(self, new_type):
        """
        Change the basic type to the new type.

        Change the basic type to the new type. In the case of a FixedSizeType the
        switch will replace the type completely, directly returning the new type.
        In the case of a homogeneous container type, a new container type will be
        returned whose underlying elements are of the new type. This method is not
        implemented for inhomogeneous containers.

        Parameters
        ----------
        new_type : PyccelType
            The new basic type.

        Returns
        -------
        PyccelType
            The new type.
        """
        assert isinstance(new_type, FixedSizeType)
        cls = type(self)
        return cls(self.element_type.switch_basic_type(new_type))

class StringType(HomogeneousContainerType, metaclass = Singleton):
    """
    Class representing Python's native string type.

    Class representing Python's native string type.
    """
    __slots__ = ()
    _name = 'str'
    _element_type = PyccelCharacterType()

    @property
    def datatype(self):
        """
        The datatype of the object.

        The datatype of the object.
        """
        return self

    def __str__(self):
        return 'str'

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

    @property
    def primitive_type(self):
        """
        The datatype category of elements of the object.

        The datatype category of elements of the object (e.g. integer, floating point).
        """
        return self

class HomogeneousTupleType(HomogeneousContainerType, TupleType, metaclass = ArgumentSingleton):
    """
    Class representing the homogeneous tuple type.

    Class representing the type of a homogeneous tuple. This
    is a container type and should be used as the class_type.

    Parameters
    ----------
    element_type : PyccelType
        The type of the elements of the homogeneous tuple.
    """
    __slots__ = ('_element_type',)

    def __init__(self, element_type):
        assert isinstance(element_type, PyccelType)
        self._element_type = element_type

    def __str__(self):
        return f'{self._name}[{self._element_type}, ...]'

class HomogeneousListType(HomogeneousContainerType, metaclass = ArgumentSingleton):
    """
    Class representing the homogeneous list type.

    Class representing the type of a homogeneous list. This
    is a container type and should be used as the class_type.

    Parameters
    ----------
    element_type : PyccelType
        The type which is stored in the homogeneous list.
    """
    __slots__ = ('_element_type',)
    _name = 'list'

    def __init__(self, element_type):
        assert isinstance(element_type, PyccelType)
        self._element_type = element_type

class HomogeneousSetType(HomogeneousContainerType, metaclass = ArgumentSingleton):
    """
    Class representing the homogeneous set type.

    Class representing the type of a homogeneous set. This
    is a container type and should be used as the class_type.

    Parameters
    ----------
    element_type : PyccelType
        The type which is stored in the homogeneous set.
    """
    __slots__ = ('_element_type',)
    _name = 'set'

    def __init__(self, element_type):
        assert isinstance(element_type, PyccelType)
        self._element_type = element_type

#==============================================================================

class CustomDataType(ContainerType, metaclass=Singleton):
    """
    Class from which user-defined types inherit.

    A general class for custom data types which is used as a
    base class when a user defines their own type using classes.
    """
    __slots__ = ()

    @property
    def datatype(self):
        """
        The datatype of the object.

        The datatype of the object.
        """
        return self

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

class InhomogeneousTupleType(ContainerType, TupleType, metaclass = ArgumentSingleton):
    """
    Class representing the inhomogeneous tuple type.

    Class representing the type of an inhomogeneous tuple. This is a
    basic datatype as it cannot be arbitrarily indexed. It is
    therefore parametrised by the datatypes that it contains.

    Parameters
    ----------
    *args : tuple of DataTypes
        The datatypes stored in the inhomogeneous tuple.
    """
    __slots__ = ('_element_types',)

    def __init__(self, *args):
        self._element_types = args
        super().__init__()

    def __str__(self):
        element_types = ', '.join(str(d) for d in self._element_types)
        return f'tuple[{element_types}]'

    def __getitem__(self, i):
        return self._element_types[i]

    def __len__(self):
        return len(self._element_types)

    def __iter__(self):
        return self._element_types.__iter__()

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
        return (self.__class__, tuple(self._element_types))

    @property
    def datatype(self):
        """
        The datatype of the object.

        The datatype of the object.
        """
        possible_types = set(t.datatype for t in self._element_types)
        if len(possible_types) == 1:
            return possible_types.pop()
        else:
            return self

class DictType(ContainerType, metaclass = ArgumentSingleton):
    """
    Class representing the homogeneous dictionary type.

    Class representing the type of a homogeneous dict. This
    is a container type and should be used as the class_type.

    Parameters
    ----------
    index_type : PyccelType
        The type of the keys of the homogeneous dictionary.
    value_type : PyccelType
        The type of the values of the homogeneous dictionary.
    """
    __slots__ = ('_index_type', '_value_type')
    _name = 'map'

    def __init__(self, index_type, value_type):
        self._index_type = index_type
        self._value_type = value_type
        super().__init__()

    def __str__(self):
        return f'map[{self._index_type.name}, {self._value_type.name}]'

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
        return (self.__class__, (self._index_type, self._value_type))

    @property
    def datatype(self):
        """
        The datatype of the object.

        The datatype of the object.
        """
        return self._index_type.datatype

#==============================================================================

def DataTypeFactory(name, argnames = (), *, BaseClass=CustomDataType):
    """
    Create a new data class.

    Create a new data class which sub-classes a DataType. This provides
    a new data type which can be used, for example, for class types.

    Parameters
    ----------
    name : str
        The name of the new class.

    argnames : list[str]
        A list of all the arguments for the new class.
        This can be used to create classes which are parametrised by a type.

    BaseClass : type inheriting from DataType
        The class from which the new type will be sub-classed.

    Returns
    -------
    type
        A new DataType class.
    """
    def class_init_func(self, **kwargs):
        """
        The __init__ function for the new CustomDataType class.
        """
        for key, value in kwargs.items():
            # here, the argnames variable is the one passed to the
            # DataTypeFactory call
            if key not in argnames:
                raise TypeError(f"Argument {key} not valid for {self.__class__.__name__}")
            setattr(self, key, value)

        BaseClass.__init__(self)

    assert isinstance(argnames, (list, tuple))
    def class_name_func(self):
        """
        The name function for the new CustomDataType class.
        """
        if argnames:
            param = ', '.join(str(getattr(self, a)) for a in argnames)
            return f'{self._name}[{param}]' #pylint: disable=protected-access
        else:
            return self._name #pylint: disable=protected-access

    newclass = type(f'Pyccel{name}', (BaseClass,),
                    {"__init__": class_init_func,
                     "name": property(class_name_func),
                     "_name": name})

    return newclass

#==============================================================================

pyccel_type_to_original_type = {
        PythonNativeBool()    : bool,
        PythonNativeInt()     : int,
        PythonNativeFloat()   : float,
        PythonNativeComplex() : complex,
        }

original_type_to_pyccel_type = {v: k for k,v in pyccel_type_to_original_type.items()}
