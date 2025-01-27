#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
Module to handle low-level language agnostic objects such as macros.
"""
from pyccel.utilities.metaclasses import ArgumentSingleton

from .basic import PyccelAstNode
from .datatypes import PyccelType

__all__ = ('IteratorType',
           'PairType',
           'MacroDefinition',
           'MacroUndef')

#------------------------------------------------------------------------------
class IteratorType(PyccelType, metaclass=ArgumentSingleton):
    """
    The type of an iterator which accesses elements of a container.

    The type of an iterator which accesses elements of a container
    (e.g. list, set, etc)

    Parameters
    ----------
    iterable_type : ContainerType
        The container that is iterated over.
    """
    __slots__ = ('_iterable_type',)
    def __init__(self, iterable_type):
        self._iterable_type = iterable_type
        super().__init__()

    @property
    def iterable_type(self):
        """
        The type of the iterable object whose elements are accessed via this type.

        The type of the iterable object whose elements are accessed via this type.
        """
        return self._iterable_type

    def __str__(self):
        return f'Iter[{self._iterable_type}]'

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

#------------------------------------------------------------------------------
class PairType(PyccelType, metaclass=ArgumentSingleton):
    """
    The type of an element of a dictionary type.

    The type of an element of a dictionary type.

    Parameters
    ----------
    key_type : PyccelType
        The type of the keys of the homogeneous dictionary.
    value_type : PyccelType
        The type of the values of the homogeneous dictionary.
    """
    __slots__ = ('_key_type', '_value_type')
    _name = 'pair'
    _container_rank = 0
    _order = None

    def __init__(self, key_type, value_type):
        self._key_type = key_type
        self._value_type = value_type
        super().__init__()

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

    def __str__(self):
        return f'pair[{self._key_type}, {self._value_type}]'

#------------------------------------------------------------------------------
class MacroDefinition(PyccelAstNode):
    """
    A class for defining a macro in a file.

    A class for defining a macro in a file.

    Parameters
    ----------
    macro_name : str
        The name of the macro.
    obj : Any
        The object that will define the macro.
    """
    _attribute_nodes = ()
    __slots__ = ('_macro_name', '_obj')

    def __init__(self, macro_name, obj):
        assert isinstance(macro_name, str)
        self._macro_name = macro_name
        self._obj = obj
        super().__init__()

    @property
    def macro_name(self):
        """
        The name of the macro being defined.

        The name of the macro being defined.
        """
        return self._macro_name

    @property
    def object(self):
        """
        The object that will define the macro.

        The object that will define the macro.
        """
        return self._obj

#------------------------------------------------------------------------------
class MacroUndef(PyccelAstNode):
    """
    A class for undefining a macro in a file.

    A class for undefining a macro in a file.

    Parameters
    ----------
    macro_name : str
        The name of the macro.
    """
    _attribute_nodes = ()
    __slots__ = ('_macro_name',)

    def __init__(self, macro_name):
        assert isinstance(macro_name, str)
        self._macro_name = macro_name
        super().__init__()

    @property
    def macro_name(self):
        """
        The name of the macro being undefined.

        The name of the macro being undefined.
        """
        return self._macro_name

