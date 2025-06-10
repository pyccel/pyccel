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
from pyccel.ast.internals import PyccelFunction

__all__ = ('ListAppend',
           'ListClear',
           'ListCopy',
           'ListExtend',
           'ListInsert',
           'ListMethod',
           'ListPop',
           'ListRemove',
           'ListReverse',
           'ListSort'
           )

#==============================================================================
class ListMethod(PyccelFunction):
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
    _attribute_nodes = PyccelFunction._attribute_nodes + ("_list_obj",)
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

    @property
    def modified_args(self):
        """
        Return a tuple of all the arguments which may be modified by this function.

        Return a tuple of all the arguments which may be modified by this function.
        This is notably useful in order to determine the constness of arguments.
        """
        return (self._list_obj,)

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
    _shape = None
    _class_type = VoidType()
    name = 'append'

    def __init__(self, list_obj, new_elem) -> None:
        expected_type = list_obj.class_type.element_type
        if new_elem.class_type != expected_type:
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
    __slots__ = ('_class_type', '_shape')
    name = 'pop'

    def __init__(self, list_obj, index_element=None) -> None:
        self._class_type = list_obj.class_type.element_type
        rank = self._class_type.rank
        self._shape = None if rank == 0 else (None,)*rank
        super().__init__(list_obj, index_element)

    @property
    def index_element(self):
        """
        The current index value for the element to be popped.

        The current index value for the element to be popped.
        """
        return self._args[0]

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
    _class_type = VoidType()
    name = 'insert'

    def __init__(self, list_obj, index, new_elem) -> None:
        if new_elem.class_type != list_obj.class_type.element_type:
            raise TypeError("Expecting an argument of the same type as the elements of the list")
        super().__init__(list_obj, index, new_elem)

    @property
    def index(self):
        """
        The index of the object after insertion in the list.

        The index of the object after insertion in the list.
        """
        return self._args[0]

    @property
    def object(self):
        """
        The object to insert into the list.

        The object to insert into the list.
        """
        return self._args[1]

#==============================================================================
class ListExtend(ListMethod):
    """
    Represents a call to the .extend() method.

    Represents a call to the .extend() method of an object with a list type,
    which adds items of an iterable (list, tuple, dictionary, etc) at the end
    of a list.
    This method is handled through the call to `_build_ListExtend` in
    the semantic stage. It then attempts to construct a `For` loop node with
    a body that calls `append()`, or direct `append()` nodes depending on
    the type of the iterable passed to `extend()`.
    This class should never be instantiated; it's only purpose is to help
    construct the annotation_method `_build_ListExtend`. 
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

#==============================================================================
class ListRemove(ListMethod) :
    """
    Represents a call to the .remove() method.
    
    Represents a call to the .remove() method which removes the first
    occurrence of a given element from the list.
    Note that the .remove() method doesn't return any value.

    >>> a = [[1, 2], [3, 4]]
    >>> a.remove([1, 2])
    >>> print(a)
    [[3, 4]]

    Parameters
    ----------
    list_obj : TypedAstNode
        The list object which the method is called from.

    removed_obj : TypedAstNode
        The object to be removed from the list.
    """
    __slots__ = ()
    _shape = None
    _class_type = VoidType()
    name = 'remove'

    def __init__(self, list_obj, removed_obj) -> None:
        if removed_obj.class_type != list_obj.class_type.element_type:
            raise TypeError(f"Can't remove an element of type {removed_obj.class_type} from {list_obj.class_type}")
        super().__init__(list_obj, removed_obj)

#==============================================================================
class ListCopy(ListMethod) :
    """
    Represents a call to the .copy() method.
    
    Represents a call to the .copy() method which is used to create a shallow
    copy of a list, meaning that any modification in the new list will be
    reflected in the original list.
    The method returns a list.

    >>> a = [1, 2, 3, 4]
    >>> b = a.copy()
    >>> print(a, b)
    [1, 2, 3, 4]
    [1, 2, 3, 4]
    >>> a[0] = 0
    >>> a[1] = 0
    >>> print(a, b)
    [0, 0, 3, 4]
    [0, 0, 3, 4]

    Parameters
    ----------
    list_obj : TypedAstNode
        The list object which the method is called from.
    """
    __slots__ = ('_class_type', '_shape')
    name = 'copy'

    def __init__(self, list_obj) -> None:
        self._shape = list_obj.shape
        self._class_type = list_obj.class_type
        super().__init__(list_obj)

    @property
    def modified_args(self):
        """
        Return a tuple of all the arguments which may be modified by this function.

        Return a tuple of all the arguments which may be modified by this function.
        This is notably useful in order to determine the constness of arguments.
        """
        return ()

#==============================================================================
class ListSort(ListMethod) :
    """
    Represents a call to the .sort() method.

    Represents a call to the `.sort()` method, which sorts the elements of the
    list in ascending order and modifies the original list in place. This means
    that the elements of the original list are rearranged to be in sorted order.
    Optional parameters are not supported, therefore they should not be provided. 
    Note that the .sort() method doesn't return any value.
    
    >>> a = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    >>> a.sort()
    >>> print(a)
    [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]

    Parameters
    ----------
    list_obj : TypedAstNode
        The list object which the method is called from.

    reverse : TypedAstNode, optional
        Argument mimicking sort's reverse parameter. This argument is 
        unsupported so it should not be provided.

    key : FunctionDef, optional
        A function to specify the sorting criteria(s). This argument is 
        unsupported so it should not be provided.
    """
    __slots__ = ()
    _shape = None
    _class_type = VoidType()
    name = 'sort'

    def __init__(self, list_obj, reverse=None, key=None) -> None:
        if reverse is not None or key is not None:
            raise TypeError("Optional Parameters are not supported for sort() method.")
        super().__init__(list_obj, reverse, key)

#==============================================================================
class ListReverse(ListMethod):
    """
    Represents a call to the .reverse() method.

    Represents a call to the `.reverse()` method, which reverses the elements of the
    list in place. This means that the elements of the original list are rearranged
    in reverse order. The .reverse() method does not return any value and does not
    accept any optional parameters.

    >>> a = [1, 2, 3, 4, 5]
    >>> a.reverse()
    >>> print(a)
    [5, 4, 3, 2, 1]

    Parameters
    ----------
    list_obj : TypedAstNode
        The list object which the method is called from.
    """
    __slots__ = ()
    _shape = None
    _class_type = VoidType()
    name = 'reverse'

    def __init__(self, list_obj) -> None:
        super().__init__(list_obj)
