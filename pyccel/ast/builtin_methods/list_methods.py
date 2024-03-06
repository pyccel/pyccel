# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
The List container has a number of built-in methods that are 
always available.

This module contains objects which describe these methods within Pyccel's AST.
"""

from pyccel.ast.datatypes import NativeVoid, NativeGeneric, NativeHomogeneousList
from pyccel.ast.internals import PyccelInternalFunction
from pyccel.utilities.stage import PyccelStage

pyccel_stage = PyccelStage()

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
    list_variable : TypedAstNode
        The list object which the method is called from.
    """
    __slots__ = ("_list_variable",)
    _attribute_nodes = ("_list_variable",)
    name = None
    def __init__(self, list_variable, *args):
        self._list_variable = list_variable
        super().__init__(*args)

    @property
    def list_variable(self):
        """
        Get the variable representing the list.

        Get the variable representing the list.
        """
        return self._list_variable

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
    list_variable : TypedAstNode
        The list object which the method is called from.
    
    new_elem : TypedAstNode
        The argument passed to append() method.
    """
    __slots__ = ()
    _dtype = NativeVoid()
    _shape = None
    _order = None
    _rank = 0
    _precision = None
    _class_type = NativeVoid()
    name = 'append'

    def __init__(self, list_variable, new_elem) -> None:
        if pyccel_stage != "syntactic":
            is_homogeneous = (
                new_elem.dtype is not NativeGeneric() and
                list_variable.dtype is not NativeGeneric() and
                list_variable.dtype == new_elem.dtype and
                list_variable.precision == new_elem.precision and
                list_variable.rank - 1 == new_elem.rank
            )
            if not is_homogeneous:
                raise TypeError("Expecting an argument of the same type as the elements of the list")
        super().__init__(list_variable, new_elem)

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
    list_variable : TypedAstNode
        The list object which the method is called from.

    index_element : TypedAstNode
        The current index value for the element to be popped.
    """
    __slots__ = ('_dtype','_precision', '_rank', '_shape', '_order')
    _class_type = NativeHomogeneousList()
    name = 'pop'

    def __init__(self, list_variable, index_element=None) -> None:
        self._rank = list_variable.rank - 1
        self._dtype = list_variable.dtype
        self._precision = list_variable.precision
        self._shape = (None if len(list_variable.shape) == 1 else tuple(list_variable.shape[1:]))
        self._order = (None if self._shape is None or len(self._shape) == 1 else list_variable.order)
        super().__init__(list_variable, index_element)

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
    list_variable : TypedAstNode
        The list object which the method is called from.
    """
    __slots__ = ()
    _dtype = NativeVoid()
    _precision = None
    _rank = 0
    _order = None
    _shape = None
    _class_type = NativeVoid()
    name = 'clear'

    def __init__(self, list_variable) -> None:
        super().__init__(list_variable)

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
    list_variable : TypedAstNode
        The list object which the method is called from.

    index : TypedAstNode
        The index value for the element to be added.
    
    new_elem : TypedAstNode
        The argument passed to insert() method.
    """
    __slots__ = ()
    _dtype = NativeVoid()
    _shape = None
    _order = None
    _rank = 0
    _precision = None
    _class_type = NativeVoid()
    name = 'insert'

    def __init__(self, list_variable, index, new_elem) -> None:
        is_homogeneous = (
            new_elem.dtype is not NativeGeneric() and
            list_variable.dtype is not NativeGeneric() and
            list_variable.dtype == new_elem.dtype and
            list_variable.precision == new_elem.precision and
            list_variable.rank - 1 == new_elem.rank
        )
        if not is_homogeneous:
            raise TypeError("Expecting an argument of the same type as the elements of the list")
        super().__init__(list_variable, index, new_elem)

#==============================================================================
class ListExtend(ListMethod):
    """
    Represents a call to the .extend() method.

    Represents a call to the .extend() method of an object with a list type,
    which adds items of an iterable (list, tuple, dictionary, etc) at the end
    of a list.
    This method returns `None`.
    The extend method is called as follows:

    >>> a = [1]
    >>> a.extend([2])
    >>> print(a)
    [1, 2]

    Parameters
    ----------
    list_variable : TypedAstNode
        The list object which the method is called from.
    
    iterable : TypedAstNode
        Iterable argument passed to extend() method.
    """
    __slots__ = ()
    name = 'extend'
