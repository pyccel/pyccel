#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
Module to handle low-level language agnostic objects such as macros.
"""
from functools import lru_cache

from .basic import PyccelAstNode, TypedAstNode
from .datatypes import PyccelType
from .variable import Variable

__all__ = ('IteratorType',
           'MacroDefinition',
           'MacroUndef',
           'ManagedMemory',
           'MemoryHandlerType',
           'PairType',
           'UnpackManagedMemory')

#------------------------------------------------------------------------------
class IteratorType(PyccelType):
    """
    The type of an iterator which accesses elements of a container.

    The type of an iterator which accesses elements of a container
    (e.g. list, set, etc)
    """
    __slots__ = ('_iterable_type',)

    @classmethod
    @lru_cache
    def get_new(cls, iterable_type):
        """
        Get the parametrised iterator type.

        Get the subclass of IteratorType describing the type of an
        iterator element of iterable_type.

        Parameters
        ----------
        iterable_type : PyccelType
            The type of the iterable object whose elements are accessed via this type.
        """
        def __init__(self):
            self._iterable_type = iterable_type
            PyccelType.__init__(self)

        return type(f'Iterator[{type(iterable_type)}]', (IteratorType,),
                    {'__init__' : __init__})()

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

#------------------------------------------------------------------------------
class PairType(PyccelType):
    """
    The type of an element of a dictionary type.

    The type of an element of a dictionary type.
    """
    __slots__ = ('_key_type', '_value_type')
    _name = 'pair'
    _container_rank = 0
    _order = None

    @classmethod
    @lru_cache
    def get_new(cls, key_type, value_type):
        """
        Get the type of an element of a dictionary type.

        Get the type of an element of a dictionary type.

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
            PyccelType.__init__(self)

        return type(f'Pair[{type(key_type)}, {type(val_type)}]', (PairType,),
                    {'__init__' : __init__})()

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
class MemoryHandlerType(PyccelType):
    """
    The type of an object which can hold a pointer and manage its memory.

    The type of an object which can hold a pointer and manage its memory by
    choosing whether or not to deallocate. This class may be used notably
    for list elements and dictionary values.
    """
    __slots__ = ('_element_type',)

    @classmethod
    @lru_cache
    def get_new(self, element_type):
        """
        Get the parametrised MemoryHandlerType.

        Get the subclass of MemoryHandlerType describing the type of an
        object which can hold a pointer and manage its memory.

        Parameters
        ----------
        element_type : PyccelType
            The type of the element whose memory is being managed.
        """
        def __init__(self):
            self._element_type = element_type
            PyccelType.__init__(self)

        return type(f'MemoryHandlerType[{type(element_type)}]', (MemoryHandlerType,),
                    {'__init__' : __init__})()

    @property
    def element_type(self):
        """
        The type of the element whose memory is being managed.

        The type of the element whose memory is being managed.
        """
        return self._element_type

    @property
    def container_rank(self):
        """
        Number of dimensions of the memory handler object.

        Number of dimensions of the memory handler object.
        This is the number of indices that can be used to
        directly index the object.
        """
        return 0

    @property
    def rank(self):
        """
        Number of dimensions of the object.

        Number of dimensions of the object. This is equal to the
        number of dimensions of the element whose memory is being
        managed.
        """
        return self._element_type.rank

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
        return shape == (() if self.rank else None)

    def __str__(self):
        return f'MemoryHandler[{self._element_type}]'

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

#------------------------------------------------------------------------------
class UnpackManagedMemory(PyccelAstNode):
    """
    Assign a pointer to a managed memory block.

    A class representing the operation whereby an object whose memory is managed
    by a MemoryHandlerType is assigned as the target of a pointer.

    Parameters
    ----------
    out_ptr : Variable
        The variable which will point at this memory block.
    managed_object : TypedAstNode
        The object whose memory is being managed.
    mem_var : Variable
        The variable responsible for managing the memory.
    """
    _attribute_nodes = ('_managed_object','_mem_var', '_out_ptr')
    __slots__ = ('_managed_object','_mem_var', '_out_ptr')

    def __init__(self, out_ptr, managed_object, mem_var):
        assert isinstance(out_ptr, Variable)
        assert isinstance(managed_object, TypedAstNode)
        assert isinstance(mem_var, Variable)
        self._managed_object = managed_object
        self._mem_var = mem_var
        self._out_ptr = out_ptr
        super().__init__()

    @property
    def out_ptr(self):
        """
        Get the variable which will point at the managed memory block.

        Get the variable which will point at the managed memory block.
        """
        return self._out_ptr

    @property
    def managed_object(self):
        """
        Get the object whose memory is being managed.

        Get the object whose memory is being managed.
        """
        return self._managed_object

    @property
    def memory_handler_var(self):
        """
        Get the variable responsible for managing the memory.

        Get the variable responsible for managing the memory.
        """
        return self._mem_var

#------------------------------------------------------------------------------
class ManagedMemory(PyccelAstNode):
    """
    A class which links a variable to the variable which manages its memory.

    A class which links a variable to the variable which manages its memory.
    This class does not need to appear in the AST description of the file.
    Simply creating an instance will add it to the AST tree which will ensure
    that it is found when examining the variable.

    Parameters
    ----------
    var : Variable
        The variable whose memory is being managed.
    mem_var : Variable
        The variable responsible for managing the memory.
    """
    __slots__ = ('_var', '_mem_var')
    _attribute_nodes = ('_var', '_mem_var')

    def __init__(self, var, mem_var):
        assert isinstance(var, Variable)
        assert isinstance(mem_var, Variable)
        assert isinstance(mem_var.class_type, MemoryHandlerType)
        self._var = var
        self._mem_var = mem_var
        super().__init__()

    @property
    def var(self):
        """
        Get the variable whose memory is being managed.

        Get the variable whose memory is being managed.
        """
        return self._var

    @property
    def mem_var(self):
        """
        Get the variable responsible for managing the memory.

        Get the variable responsible for managing the memory.
        """
        return self._mem_var
