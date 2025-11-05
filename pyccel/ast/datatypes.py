# coding: utf-8
#pylint: disable=no-member, protected-access

#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#

"""
Classes and methods that handle supported datatypes in C/Fortran.
"""
from functools import lru_cache

import numpy

from pyccel.utilities.metaclasses import Singleton
from .basic import iterable

__all__ = (
        # ------------ Super classes ------------
        'ContainerType',
        'FixedSizeType',
        'PrimitiveType',
        'PyccelType',
        # ------------ Primitive types ------------
        'PrimitiveBooleanType',
        'PrimitiveCharacterType',
        'PrimitiveComplexType',
        'PrimitiveFloatingPointType',
        'PrimitiveIntegerType',
        # ------------ Modifying types ------------
        'FinalType',
        # ------------ Fixed size types ------------
        'CharType',
        'FixedSizeNumericType',
        'GenericType',
        'PythonNativeBool',
        'PythonNativeComplex',
        'PythonNativeFloat',
        'PythonNativeInt',
        'PythonNativeNumericType',
        'SymbolicType',
        'TypeAlias',
        'VoidType',
        # ------------ Container types ------------
        'CustomDataType',
        'DictType',
        'HomogeneousContainerType',
        'HomogeneousListType',
        'HomogeneousSetType',
        'HomogeneousTupleType',
        'InhomogeneousTupleType',
        'StringType',
        'TupleType',
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

    def __init__(self): #pylint: disable=useless-parent-delegation
        # This __init__ function is required so the Singleton can
        # always detect a signature
        super().__init__()

    def __str__(self):
        return self._name

class PrimitiveBooleanType(PrimitiveType):
    """
    Class representing a boolean datatype.

    Class representing a boolean datatype.
    """
    __slots__ = ()
    _name = 'boolean'

class PrimitiveIntegerType(PrimitiveType):
    """
    Class representing an integer datatype.

    Class representing an integer datatype.
    """
    __slots__ = ()
    _name = 'integer'

class PrimitiveFloatingPointType(PrimitiveType):
    """
    Class representing a floating point datatype.

    Class representing a floating point datatype.
    """
    __slots__ = ()
    _name = 'floating point'

class PrimitiveComplexType(PrimitiveType):
    """
    Class representing a complex datatype.

    Class representing a complex datatype.
    """
    __slots__ = ()
    _name = 'complex'

class PrimitiveCharacterType(PrimitiveType):
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

    A type also contains an attribute _name which can be useful to examine
    the type.
    """
    __slots__ = ()

    @property
    def name(self):
        """
        Get the name of the pyccel type.
        
        Get the name of the pyccel type.
        """
        return self._name

    def __init__(self): #pylint: disable=useless-parent-delegation
        # This __init__ function is required so the Singleton can
        # always detect a signature
        super().__init__()

    def __str__(self):
        return self._name

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

    def shape_is_compatible(self, shape):
        """
        Check if the provided shape is compatible with the datatype.

        Check if the provided shape is compatible with the format expected for
        this datatype.

        Parameters
        ----------
        shape : Any
            The proposed shape.

        Returns
        -------
        bool
            True if the shape is acceptable, False otherwise.
        """
        return shape is None

#==============================================================================
class FinalType:
    """
    A class to get PyccelType subclasses describing constant values.

    A class to get PyccelType subclasses describing constant values.
    """
    __slots__ = ()

    @classmethod
    @lru_cache
    def get_new(cls, underlying_type):
        """
        Get the parameterised Final type.

        Get the parameterised Final type Final[underlying_type].

        Parameters
        ----------
        underlying_type : PyccelType
            The type which is characterised as final.
        """
        if isinstance(underlying_type, FinalType):
            return underlying_type

        type_class = type(underlying_type)
        def __init__(self):
            self._underlying_type = underlying_type
            type(underlying_type).__init__(self)
        def __hash__(self):
            return type_class.__hash__(underlying_type)
        def __eq__(self, other):
            return type_class.__eq__(underlying_type, other)
        def get_underlying_type(self):
            """
            Get the type that is indicated as const.

            Get the type that is indicated as const.
            """
            return self._underlying_type
        return type(f'Final[{type_class.__name__}]', (FinalType, type_class,),
                    {'__init__' : __init__,
                     '__hash__' : __hash__,
                     '__eq__' : __eq__,
                     'underlying_type': property(get_underlying_type)})()

    def __str__(self):
        return f'Final[{self._underlying_type}]'

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
        return self._primitive_type

    @property
    def rank(self):
        """
        Number of dimensions of the object.

        Number of dimensions of the object. If the object is a scalar then
        this is equal to 0.
        """
        return 0

    @property
    def order(self):
        """
        The data layout ordering in memory.

        Indicates whether the data is stored in row-major ('C') or column-major
        ('F') format. This is only relevant if rank > 1. When it is not relevant
        this function returns None.
        """
        return None

    def switch_basic_type(self, new_type):
        """
        Change the basic type to the new type.

        Change the basic type to the new type. In the case of a FixedSizeType the
        switch will replace the type completely, directly returning the new type.

        Parameters
        ----------
        new_type : FixedSizeType
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
        number of bytes that the datatype takes up in memory. For basic types the
        number is equivalent to the number of bytes in memory (e.g. `float64` has
        precision = 8 as it takes up 8 bytes), however for less simple types the
        connection is less trivial. For example `complex128` has precision = 8 as
        it is comprised of two `float64` objects (which have precision=8).
        It should be noted that this is not the convention chosen by NumPy (in NumPy
        a `complex128` is so named because `16*8=precision*bits_in_a_byte=128`).

        The precision in Pyccel is equivalent to the `kind` parameter in Fortran.
        """
        return self._precision

class PythonNativeNumericType(FixedSizeNumericType):
    """
    Base class representing a built-in scalar numeric datatype.

    Base class representing a built-in scalar numeric datatype.
    """
    __slots__ = ()

class PythonNativeBool(PythonNativeNumericType):
    """
    Class representing Python's native boolean type.

    Class representing Python's native boolean type.
    """
    __slots__ = ()
    _name = 'bool'
    _primitive_type = PrimitiveBooleanType()
    _precision = -1

    @lru_cache
    def __add__(self, other):
        if isinstance(other, PythonNativeBool):
            return PythonNativeInt()
        elif isinstance(other, PythonNativeNumericType):
            return other
        else:
            return NotImplemented

    @lru_cache
    def __and__(self, other):
        if isinstance(other, PythonNativeBool):
            return PythonNativeBool()
        elif isinstance(other, PythonNativeNumericType):
            return other
        else:
            return NotImplemented

class PythonNativeInt(PythonNativeNumericType):
    """
    Class representing Python's native integer type.

    Class representing Python's native integer type.
    """
    __slots__ = ()
    _name = 'int'
    _primitive_type = PrimitiveIntegerType()
    _precision = numpy.dtype(int).alignment

    @lru_cache
    def __add__(self, other):
        if isinstance(other, PythonNativeBool):
            return self
        elif isinstance(other, PythonNativeNumericType):
            return other
        else:
            return NotImplemented

    @lru_cache
    def __and__(self, other):
        if isinstance(other, PythonNativeNumericType):
            return self
        else:
            return NotImplemented


class PythonNativeFloat(PythonNativeNumericType):
    """
    Class representing Python's native floating point type.

    Class representing Python's native floating point type.
    """
    __slots__ = ()
    _name = 'float'
    _primitive_type = PrimitiveFloatingPointType()
    _precision = 8

    @lru_cache
    def __add__(self, other):
        if isinstance(other, PythonNativeComplex):
            return other
        elif isinstance(other, PythonNativeNumericType):
            return self
        else:
            return NotImplemented


class PythonNativeComplex(PythonNativeNumericType):
    """
    Class representing Python's native complex type.

    Class representing Python's native complex type.
    """
    __slots__ = ('_element_type',)
    _name = 'complex'
    _primitive_type = PrimitiveComplexType()
    _precision = 8

    @lru_cache
    def __add__(self, other):
        if isinstance(other, PythonNativeNumericType):
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

    def __eq__(self, other):
        return True

    def __hash__(self):
        return hash(self.__class__)

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
    _primitive_type = PrimitiveCharacterType()

#==============================================================================
class TypeAlias(SymbolicType):
    """
    Class representing the type of a symbolic object describing a type descriptor.

    Class representing the type of a symbolic object describing a type descriptor.
    This type is equivalent to Python's built-in typing.TypeAlias.

    See Also
    --------
    typing.TypeAlias :
        See documentation of `typing.TypeAlias`: https://docs.python.org/3/library/typing.html#typing.TypeAlias .
    """
    __slots__ = ()
    _name = 'TypeAlias'

#==============================================================================

class ContainerType(PyccelType):
    """
    Base class representing a type which contains objects of other types.

    Base class representing a type which contains objects of other types.
    E.g. classes, arrays, etc.
    """
    __slots__ = ()

    def shape_is_compatible(self, shape):
        """
        Check if the provided shape is compatible with the datatype.

        Check if the provided shape is compatible with the format expected for
        this datatype.

        Parameters
        ----------
        shape : Any
            The proposed shape.

        Returns
        -------
        bool
            True if the shape is acceptable, False otherwise.
        """
        return isinstance(shape, tuple) and len(shape) == self.container_rank

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
        Precision of the datatype of the object.

        The precision of the datatype of the object. This number is related to the
        number of bytes that the datatype takes up in memory. For basic types the
        number is equivalent to the number of bytes in memory (e.g. `float64` has
        precision = 8 as it takes up 8 bytes), however for less simple types the
        connection is less trivial. For example `complex128` has precision = 8 as
        it is comprised of two `float64` objects (which have precision=8).
        It should be noted that this is not the convention chosen by NumPy (in NumPy
        a `complex128` is so named because `16*8=precision*bits_in_a_byte=128`).

        The precision in Pyccel is equivalent to the `kind` parameter in Fortran.
        """
        return self.element_type.precision

    @property
    def element_type(self):
        """
        The type of elements of the object.

        The PyccelType describing an element of the container.
        """
        return self._element_type

    def __str__(self):
        return f'{self._name}[{self._element_type}]'

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
        new_type : FixedSizeType
            The new basic type.

        Returns
        -------
        PyccelType
            The new type.
        """
        assert isinstance(new_type, FixedSizeType)
        cls = type(self)
        return cls.get_new(self.element_type.switch_basic_type(new_type))

    def switch_rank(self, new_rank, new_order = None):
        """
        Get a type which is identical to this type in all aspects except the rank.

        Get a type which is identical to this type in all aspects except the rank.
        The order must be provided if the rank is increased from 1. This is never
        the case for 1D containers.

        Parameters
        ----------
        new_rank : int
            The rank of the new type.

        new_order : str, optional
            The order of the new type. For 1D containers this should not be provided.

        Returns
        -------
        PyccelType
            The new type.
        """
        assert new_order is None
        rank = self.rank
        assert new_rank < rank

        if new_rank == rank:
            return self
        elif rank - new_rank == self.container_rank:
            return self.element_type
        else:
            return self.element_type.switch_rank(new_rank - self.container_rank)

    @property
    def container_rank(self):
        """
        Number of dimensions of the container.

        Number of dimensions of the object described by the container. This is
        equal to the number of values required to index an element of this container.
        """
        return self._container_rank

    @property
    def rank(self):
        """
        Number of dimensions of the object.

        Number of dimensions of the object. If the object is a scalar then
        this is equal to 0.
        """
        return self.container_rank + self.element_type.rank

    @property
    def order(self):
        """
        The data layout ordering in memory.

        Indicates whether the data is stored in row-major ('C') or column-major
        ('F') format. This is only relevant if rank > 1. When it is not relevant
        this function returns None.
        """
        return self._order

class StringType(ContainerType, metaclass = Singleton):
    """
    Class representing Python's native string type.

    Class representing Python's native string type.
    """
    __slots__ = ()
    _name = 'str'

    @property
    def datatype(self):
        """
        The datatype of the object.

        The datatype of the object.
        """
        return self

    def __str__(self):
        return 'str'

    @property
    def primitive_type(self):
        """
        The datatype category of elements of the object.

        The datatype category of elements of the object (e.g. integer, floating point).
        """
        return self

    @property
    def rank(self):
        """
        Number of dimensions of the object.

        Number of dimensions of the object. If the object is a scalar then
        this is equal to 0.
        """
        return 1

    @property
    def container_rank(self):
        """
        Number of dimensions of the container.

        Number of dimensions of the object described by the container. This is
        equal to the number of values required to index an element of this container.
        """
        return 1

    @property
    def order(self):
        """
        The data layout ordering in memory.

        Indicates whether the data is stored in row-major ('C') or column-major
        ('F') format. This is only relevant if rank > 1. When it is not relevant
        this function returns None.
        """
        return None

    @property
    def element_type(self):
        """
        The type of elements of the object.

        The PyccelType describing an element of the container.
        """
        return CharType()

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __hash__(self):
        return hash(self.__class__)

class HomogeneousTupleType(HomogeneousContainerType, TupleType, metaclass = Singleton):
    """
    Class representing the homogeneous tuple type.

    Class representing the type of a homogeneous tuple. This
    is a container type and should be used as the class_type.
    """
    _name = 'tuple'
    __slots__ = ('_element_type', '_order')
    _container_rank = 1

    @classmethod
    @lru_cache
    def get_new(cls, element_type):
        """
        Get the parametrised homogeneous tuple type.

        Get the subclass of HomogeneousTupleType describing the
        type of a tuple[element_type, ...].

        Parameters
        ----------
        element_type : PyccelType
            The type of the elements of the homogeneous tuple.
        """
        def __init__(self):
            self._element_type = element_type
            self._order = 'C' if (element_type.order == 'C' or element_type.rank == 1) else None
            HomogeneousContainerType.__init__(self)

        return type(f'HomogeneousTuple[{type(element_type)}, ...]', (HomogeneousTupleType,),
                    {'__init__' : __init__})()

    def __str__(self):
        return f'tuple[{self._element_type}, ...]'

    def shape_is_compatible(self, shape):
        """
        Check if the provided shape is compatible with the datatype.

        Check if the provided shape is compatible with the format expected for
        this datatype.

        Parameters
        ----------
        shape : Any
            The proposed shape.

        Returns
        -------
        bool
            True if the shape is acceptable, False otherwise.
        """
        # TODO: Remove this specialisation if tuples are saved in lists instead of ndarrays
        return isinstance(shape, tuple) and len(shape) == self.rank

    def __eq__(self, other):
        return isinstance(other, HomogeneousTupleType) and self.element_type == other.element_type

    def __hash__(self):
        return hash((HomogeneousTupleType, self.element_type))

class HomogeneousListType(HomogeneousContainerType, metaclass = Singleton):
    """
    Class representing the homogeneous list type.

    Class representing the type of a homogeneous list. This
    is a container type and should be used as the class_type.
    """
    __slots__ = ('_element_type', '_order')
    _name = 'list'
    _container_rank = 1

    @classmethod
    @lru_cache
    def get_new(cls, element_type):
        """
        Get the parametrised homogeneous list type.

        Get the subclass of HomogeneousListType describing the
        type of a list[element_type].

        Parameters
        ----------
        element_type : PyccelType
            The type which is stored in the homogeneous list.
        """
        def __init__(self):
            self._element_type = element_type
            self._order = 'C' if (element_type.order == 'C' or element_type.rank == 1) else None
            HomogeneousContainerType.__init__(self)

        return type(f'HomogeneousList[{type(element_type)}]', (HomogeneousListType,),
                    {'__init__' : __init__})()

    def __eq__(self, other):
        return isinstance(other, HomogeneousListType) and self._element_type == other._element_type \
                and self._order == other._order

    def __hash__(self):
        return hash((HomogeneousListType, self._element_type, self._order))

class HomogeneousSetType(HomogeneousContainerType, metaclass = Singleton):
    """
    Class representing the homogeneous set type.

    Class representing the type of a homogeneous set. This
    is a container type and should be used as the class_type.
    """
    __slots__ = ('_element_type',)
    _name = 'set'
    _container_rank = 1
    _order = None

    @classmethod
    @lru_cache
    def get_new(cls, element_type):
        """
        Get the parametrised homogeneous set type.

        Get the subclass of HomogeneousSetType describing the
        type of a set[element_type].

        Parameters
        ----------
        element_type : PyccelType
            The type which is stored in the homogeneous set.
        """
        def __init__(self):
            self._element_type = element_type
            HomogeneousContainerType.__init__(self)

        return type(f'HomogeneousSet[{type(element_type)}]', (HomogeneousSetType,),
                    {'__init__' : __init__})()

    def __eq__(self, other):
        return isinstance(other, HomogeneousSetType) and self._element_type == other._element_type

    def __hash__(self):
        return hash((HomogeneousSetType, self._element_type))

#==============================================================================

class CustomDataType(PyccelType, metaclass=Singleton):
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

    @property
    def rank(self):
        """
        Number of dimensions of the object.

        Number of dimensions of the object. If the object is a scalar then
        this is equal to 0.
        """
        return 0

    @property
    def order(self):
        """
        The data layout ordering in memory.

        Indicates whether the data is stored in row-major ('C') or column-major
        ('F') format. This is only relevant if rank > 1. When it is not relevant
        this function returns None.
        """
        return None

class InhomogeneousTupleType(ContainerType, TupleType, metaclass = Singleton):
    """
    Class representing the inhomogeneous tuple type.

    Class representing the type of an inhomogeneous tuple. This is a
    basic datatype as it cannot be arbitrarily indexed. It is
    therefore parametrised by the datatypes that it contains.
    """
    __slots__ = ('_element_types', '_datatype', '_container_rank', '_order')

    @classmethod
    @lru_cache
    def get_new(cls, *args):
        """
        Get the parametrised inhomogeneous tuple type.

        Get the parametrised inhomogeneous tuple type.

        Parameters
        ----------
        *args : tuple of DataTypes
            The datatypes stored in the inhomogeneous tuple.
        """
        # Determine datatype
        possible_types = set(t.datatype for t in args)
        try:
            dtype = possible_types.pop()
        except KeyError:
            dtype = GenericType()

        # Determine rank
        elem_ranks = set(elem.rank for elem in args)
        if len(elem_ranks) == 1:
            container_rank = elem_ranks.pop() + 1
        else:
            container_rank = 1

        # Determine order
        if container_rank == 2:
            order = 'C'
        elif container_rank > 2:
            elem_orders = set(elem.order for elem in args)
            if len(elem_orders) == 1 and elem_orders.pop() == 'C':
                order = 'C'
            else:
                order = None
        else:
            order = None

        def __init__(self):
            self._element_types = args

            self._datatype = dtype if all(d == dtype for d in possible_types) else self
            self._container_rank = container_rank
            self._order = order
            ContainerType.__init__(self)

        name = 'InhomogeneousTuple[' + ', '.join(str(type(e)) for e in args) + ']'

        return type(name, (InhomogeneousTupleType,),
                    {'__init__' : __init__})()

    def __str__(self):
        element_types = ', '.join(str(d) for d in self._element_types)
        return f'tuple[{element_types}]'

    def __getitem__(self, i):
        return self._element_types[i]

    def __len__(self):
        return len(self._element_types)

    def __iter__(self):
        return self._element_types.__iter__()

    @property
    def datatype(self):
        """
        The datatype of the object.

        The datatype of the object. For an inhomogeneous tuple the datatype is the type
        of the tuple unless the tuple is comprised of containers which are all based on
        compatible data types. In this case one of the underlying types is returned.
        """
        return self._datatype

    @property
    def container_rank(self):
        """
        Number of dimensions of the container.

        Number of dimensions of the object described by the container. This is
        equal to the number of values required to index an element of this container.
        """
        return 1

    @property
    def rank(self):
        """
        Number of dimensions of the object.

        Number of dimensions of the object. If the object is a scalar then
        this is equal to 0.
        """
        return self._container_rank

    @property
    def order(self):
        """
        The data layout ordering in memory.

        Indicates whether the data is stored in row-major ('C') or column-major
        ('F') format. This is only relevant if rank > 1. When it is not relevant
        this function returns None.
        """
        return self._order

    def shape_is_compatible(self, shape):
        """
        Check if the provided shape is compatible with the datatype.

        Check if the provided shape is compatible with the format expected for
        this datatype.

        Parameters
        ----------
        shape : Any
            The proposed shape.

        Returns
        -------
        bool
            True if the shape is acceptable, False otherwise.
        """
        return super().shape_is_compatible(shape) and shape[0] == len(self._element_types)

class DictType(ContainerType, metaclass = Singleton):
    """
    Class representing the homogeneous dictionary type.

    Class representing the type of a homogeneous dict. This
    is a container type and should be used as the class_type.
    """
    __slots__ = ('_key_type', '_value_type')
    _name = 'dict'
    _container_rank = 1
    _order = None

    @classmethod
    @lru_cache
    def get_new(cls, key_type, value_type):
        """
        Get the parametrised dictionary type.

        Get the subclass describing a dictionary type
        Dict[key_type, value_type].

        Parameters
        ----------
        key_type : PyccelType
            The type of the keys of the homogeneous dictionary.
        value_type : PyccelType
            The type of the values of the homogeneous dictionary.
        """
        def __init__(self):
            self._key_type = key_type
            self._value_type = value_type
            ContainerType.__init__(self)

        return type(f'Dict[{type(key_type)}, {type(value_type)}]', (DictType,),
                    {'__init__' : __init__})()

    def __str__(self):
        return f'dict[{self._key_type}, {self._value_type}]'

    @property
    def datatype(self):
        """
        The datatype of the object.

        The datatype of the object.
        """
        return self._key_type.datatype

    @property
    def key_type(self):
        """
        The type of the keys of the object.

        The type of the keys of the object.
        """
        return self._key_type

    @property
    def value_type(self):
        """
        The type of the values of the object.

        The type of the values of the object.
        """
        return self._value_type

    @property
    def container_rank(self):
        """
        Number of dimensions of the container.

        Number of dimensions of the object described by the container. This is
        equal to the number of values required to index an element of this container.
        """
        return 1

    @property
    def rank(self):
        """
        Number of dimensions of the object.

        Number of dimensions of the object. If the object is a scalar then
        this is equal to 0.
        """
        return self._container_rank + self._value_type.rank

    @property
    def order(self):
        """
        The data layout ordering in memory.

        Indicates whether the data is stored in row-major ('C') or column-major
        ('F') format. This is only relevant if rank > 1. When it is not relevant
        this function returns None.
        """
        return None

    def __eq__(self, other):
        return isinstance(other, DictType) and self.key_type == other.key_type \
                and self.value_type == other.value_type

    def __hash__(self):
        return hash((DictType, self._key_type, self._value_type))

#==============================================================================

def DataTypeFactory(ll_name, python_name, argnames = (), *, BaseClass=CustomDataType):
    """
    Create a new data class.

    Create a new data class which sub-classes a DataType. This provides
    a new data type which can be used, for example, for class types.

    Parameters
    ----------
    ll_name : str
        The low-level name of the new class.

    python_name : str
        The original name of the new class matching the name used in Python.

    argnames : iterable[str]
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

        BaseClass.__init__(self) # pylint: disable=unnecessary-dunder-call

    assert iterable(argnames)
    assert all(isinstance(a, str) for a in argnames)

    def class_name_func(self):
        """
        The name function for the new CustomDataType class.
        """
        if argnames:
            param = ', '.join(str(getattr(self, a)) for a in argnames)
            return f'{self._name}[{param}]' #pylint: disable=protected-access
        else:
            return self._name #pylint: disable=protected-access

    def low_level_name(self):
        """
        The low_level_name function for the new CustomDataType class.
        This describes the name that will be used in the low-level language.
        """
        return ll_name

    newclass = type(f'Pyccel{python_name}', (BaseClass,),
                    {"__init__": class_init_func,
                     "name": property(class_name_func),
                     "_name": python_name,
                     "low_level_name": property(low_level_name)})

    return newclass

#==============================================================================

pyccel_type_to_original_type = {
        PythonNativeBool()    : bool,
        PythonNativeInt()     : int,
        PythonNativeFloat()   : float,
        PythonNativeComplex() : complex,
        }

original_type_to_pyccel_type = {v: k for k,v in pyccel_type_to_original_type.items()}
