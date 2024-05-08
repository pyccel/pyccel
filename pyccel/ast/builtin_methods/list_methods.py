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

from pyccel.ast.datatypes import NativeVoid, NativeGeneric, NativeHomogeneousList
from pyccel.ast.internals import PyccelInternalFunction


__all__ = ('ListAppend',
           'ListClear',
           'ListInsert',
           'ListPop',
           )

#==============================================================================
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
    
    new_elem : TypedAstNode
        The argument passed to append() method.
    """
    __slots__ = ("_list_variable", "_append_arg")
    _attribute_nodes = ("_list_variable", "_append_arg")
    _dtype = NativeVoid()
    _shape = None
    _order = None
    _rank = 0
    _precision = -1
    _class_type = NativeVoid()
    name = 'append'

    def __init__(self, list_variable, new_elem) -> None:
        is_homogeneous = (
            new_elem.dtype is not NativeGeneric() and
            list_variable.dtype is not NativeGeneric() and
            list_variable.dtype == new_elem.dtype and
            list_variable.precision == new_elem.precision and
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
    __slots__ = ('_dtype','_precision', '_index','_list_variable')
    _attribute_nodes = ('_index','_list_variable')
    _rank = 0
    _order = None
    _shape = None
    _class_type = NativeHomogeneousList()
    name = 'pop'

    def __init__(self, list_variable, index_element=None):
        self._index = index_element
        self._list_variable = list_variable
        self._dtype = list_variable.dtype
        self._precision = list_variable.precision
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
    _dtype = NativeVoid()
    _precision = -1
    _rank = 0
    _order = None
    _shape = None
    _class_type = NativeVoid()
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

#==============================================================================
class ListInsert(PyccelInternalFunction):
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
    list_variable : Variable
        The variable representing the list.

    index : TypedAstNode
        The index value for the element to be added.
    
    new_elem : TypedAstNode
        The argument passed to insert() method.
    """
    __slots__ = ("_index", "_list_variable", "_insert_arg")
    _attribute_nodes = ("_index", "_list_variable", "_insert_arg")
    _dtype = NativeVoid()
    _shape = None
    _order = None
    _rank = 0
    _precision = -1
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
        self._index = index
        self._list_variable = list_variable
        self._insert_arg = new_elem
        super().__init__()

    @property
    def index(self):
        """
        Index in which the element will be added.

        Index in which the element will be added.
        """
        return self._index

    @property
    def list_variable(self):
        """
        Get the variable representing the list.

        Get the variable representing the list.
        """
        return self._list_variable

    @property
    def insert_argument(self):
        """
        Get the argument which is passed to insert().

        Get the argument which is passed to insert().
        """
        return self._insert_arg
