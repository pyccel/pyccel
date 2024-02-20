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

from pyccel.ast.datatypes import VoidType, GenericType, HomogeneousListType
from pyccel.ast.internals import PyccelInternalFunction


__all__ = ('ListAppend',
           'ListClear',
           'ListPop'
           )


class ListAppend(PyccelInternalFunction):
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
    list_variable : Variable
        The variable representing the list.
    
    new_elem : Variable
        The argument passed to append() method.
    """
    __slots__ = ("_list_variable", "_append_arg")
    _attribute_nodes = ("_list_variable", "_append_arg")
    _dtype = VoidType()
    _shape = None
    _order = None
    _rank = 0
    _class_type = VoidType()
    name = 'append'

    def __init__(self, list_variable, new_elem) -> None:
        is_homogeneous = (
            new_elem.dtype is not GenericType() and
            list_variable.dtype is not GenericType() and
            list_variable.dtype == new_elem.dtype and
            list_variable.rank - 1 == new_elem.rank
        )
        if not is_homogeneous:
            raise TypeError("Expecting an argument of the same type as the elements of the list")
        self._list_variable = list_variable
        self._append_arg = new_elem
        super().__init__()

    @property
    def list_variable(self):
        """
        Get the variable representing the list.

        Get the variable representing the list.
        """
        return self._list_variable

    @property
    def append_argument(self):
        """
        Get the argument which is passed to append().

        Get the argument which is passed to append().
        """
        return self._append_arg

#==============================================================================
class ListPop(PyccelInternalFunction) :
    """
    Represents a call to the .pop() method.
    
    Represents a call to the .pop() method which
    removes the item at the specified index. 
    The method also returns the removed item.

    Parameters
    ----------
    list_variable : TypedAstNode
        The name of the list.

    index_element : TypedAstNode
        The current index value for the element to be popped.
    """
    __slots__ = ('_dtype','_class_type', '_index','_list_variable')
    _attribute_nodes = ('_index','_list_variable')
    _rank = 0
    _order = None
    _shape = None
    name = 'pop'

    def __init__(self, list_variable, index_element=None):
        self._index = index_element
        self._list_variable = list_variable
        self._dtype = list_variable.dtype
        self._class_type = HomogeneousListType(self._dtype)
        super().__init__()

    @property
    def pop_index(self):
        """
        The current index value for the element to be popped.

        The current index value for the element to be popped.
        """
        return self._index

    @property
    def list_variable(self):
        """
        Provide the name of the list as the return value.
        
        Provide the name of the list as the return value.
        """
        return self._list_variable

#==============================================================================
class ListClear(PyccelInternalFunction) :
    """
    Represents a call to the .clear() method.
    
    Represents a call to the .clear() method which deletes all elements from a list, 
    effectively turning it into an empty list.
    Note that the .clear() method doesn't return any value.

    Parameters
    ----------
    list_variable : TypedAstNode
        The name of the list.
    """
    __slots__ = ('_list_variable',)
    _attribute_nodes = ('_list_variable',)
    _dtype = VoidType()
    _rank = 0
    _order = None
    _shape = None
    _class_type = VoidType()
    name = 'clear'

    def __init__(self, list_variable):
        self._list_variable = list_variable
        super().__init__()

    @property
    def list_variable(self):
        """
        Provide the name of the list as the return value.

        Provide the name of the list as the return value.
        """
        return self._list_variable
