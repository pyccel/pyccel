# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
The List container has a number of built-in methods that are 
always available.

This module contains objects which describe these methods within Pyccel's AST.
"""

from pyccel.ast.datatypes import VoidType
from pyccel.ast.internals import PyccelInternalFunction

__all__ = ('ListAppend',
           'ListClear',
           'ListExtend',
           'ListInsert',
           'ListMethod',
           'ListPop',
           )

#==============================================================================
class ListMethod(PyccelInternalFunction):
    """
    Abstract class for list method calls.

    A subclass of this base class represents calls to a specific list
    method.

    Parameters
    ----------
    list_obj : TypedAstNode
        The object which the method is called from.
    
    *args : TypedAstNode
        The arguments passed to list methods.
    """
    __slots__ = ("_list_obj",)
    _attribute_nodes = ("_list_obj",)
    name = None
    def __init__(self, list_obj, *args):
        self._list_obj = list_obj
        super().__init__(*args)

    @property
    def list_obj(self):
        """
        Get the object representing the list.

        Get the object representing the list.
        """
        return self._list_obj

#==============================================================================
class ListAppend(ListMethod):
    """
    Represents a call to the .append() method.

    Represents a call to the .append() method of an object with a list type,
    which adds an element to the end of the list. This method returns `None`.
    The append method is called as follows:

    >>> a = [1]
    >>> a.append(2)
    >>> print(a)
    [1, 2]

    Parameters
    ----------
    list_obj : TypedAstNode
        The list object which the method is called from.
    
    new_elem : TypedAstNode
        The argument passed to append() method.
    """
    __slots__ = ()
    _dtype = VoidType()
    _shape = None
    _order = None
    _rank = 0
    _class_type = VoidType()
    name = 'append'

    def __init__(self, list_obj, new_elem) -> None:
        expected_type = list_obj.class_type.element_type
        is_homogeneous = (
            new_elem.class_type == expected_type and
            list_obj.rank - 1 == new_elem.rank
        )
        if not is_homogeneous:
            raise TypeError(f"Expecting an argument of the same type as the elements of the list ({expected_type}) but received {new_elem.class_type}")
        super().__init__(list_obj, new_elem)

#==============================================================================
class ListPop(ListMethod) :
    """
    Represents a call to the .pop() method.
    
    Represents a call to the .pop() method which
    removes the item at the specified index. 
    The method also returns the removed item.

    >>> [1, 2].pop()
    2

    Parameters
    ----------
    list_obj : TypedAstNode
        The list object which the method is called from.

    index_element : TypedAstNode
        The current index value for the element to be popped.
    """
    __slots__ = ('_class_type', '_rank', '_shape', '_order')
    name = 'pop'

    def __init__(self, list_obj, index_element=None) -> None:
        self._rank = list_obj.rank - 1
        self._shape = (None if len(list_obj.shape) == 1 else tuple(list_obj.shape[1:]))
        self._order = (None if self._shape is None or len(self._shape) == 1 else list_obj.order)
        self._class_type = list_obj.class_type.element_type
        super().__init__(list_obj, index_element)

#==============================================================================
class ListClear(ListMethod) :
    """
    Represents a call to the .clear() method.
    
    Represents a call to the .clear() method which deletes all elements from a list, 
    effectively turning it into an empty list.
    Note that the .clear() method doesn't return any value.

    >>> a = [1, 2]
    >>> a.clear()
    >>> print(a)
    []

    Parameters
    ----------
    list_obj : TypedAstNode
        The list object which the method is called from.
    """
    __slots__ = ()
    _rank = 0
    _order = None
    _shape = None
    _class_type = VoidType()
    name = 'clear'

    def __init__(self, list_obj) -> None:
        super().__init__(list_obj)

#==============================================================================
class ListInsert(ListMethod):
    """
    Represents a call to the .insert() method.

    Represents a call to the .insert() method of an object with a list type,
    which inserts a given element at a given index in a list.
    This method returns `None`.
    The insert method is called as follows:

    >>> a = [2, 3, 4]
    >>> a.insert(0, 1)
    >>> print(a)
    [1, 2, 3, 4]

    Parameters
    ----------
    list_obj : TypedAstNode
        The list object which the method is called from.

    index : TypedAstNode
        The index value for the element to be added.
    
    new_elem : TypedAstNode
        The argument passed to insert() method.
    """
    __slots__ = ()
    _shape = None
    _order = None
    _rank = 0
    _class_type = VoidType()
    name = 'insert'

    def __init__(self, list_obj, index, new_elem) -> None:
        expected_type = list_obj.class_type.element_type
        is_homogeneous = (
            new_elem.class_type == expected_type and
            list_obj.rank - 1 == new_elem.rank
        )
        if not is_homogeneous:
            raise TypeError("Expecting an argument of the same type as the elements of the list")
        super().__init__(list_obj, index, new_elem)

#==============================================================================
class ListExtend(ListMethod):
    """
    Represents a call to the .extend() method.

    Represents a call to the .extend() method of an object with a list type,
    which adds items of an iterable (list, tuple, dictionary, etc) at the end
    of a list.
    This method is handled through the call to `_visit_ListExtend` in
    the semantic stage. It then attempts to construct a `For` loop node with
    a body that calls `append()`, or direct `append()` nodes depending on
    the type of the iterable passed to `extend()`.
    This class should never be instantiated; it's only purpose is to help
    construct the annotation_method `_visit_ListExtend`. 
    The extend method is called as follows:

    >>> a = [1, 2, 3]
    >>> a.extend(range(4, 8))
    >>> print(a)
    [1, 2, 3, 4, 5, 6, 7]

    Parameters
    ----------
    list_obj : TypedAstNode
        The list object which the method is called from.

    iterable : TypedAstNode
        The argument passed to extend() method.
    """
    __slots__ = ()
    name = 'extend'

    def __init__(self, list_obj, iterable) -> None:
        super().__init__(list_obj, iterable)
