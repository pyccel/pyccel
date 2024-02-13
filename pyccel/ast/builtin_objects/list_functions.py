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
    _dtype = NativeVoid()
    _shape = None
    _order = None
    _rank = 0
    _precision = -1
    _class_type = NativeHomogeneousList()
    name = 'append'

    def __init__(self, list_variable, new_elem) -> None:
        conditions = (
            new_elem.dtype is not NativeGeneric() and
            list_variable.dtype == new_elem.dtype and
            list_variable.precision == new_elem.precision and
            list_variable.rank - 1 == new_elem.rank
        )
        is_homogeneous = list_variable.dtype is not NativeGeneric() and conditions
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

class ListInsert(PyccelInternalFunction):
    """
    Represents a call to the .insert() method.

    Represents a call to the .insert() method of an object with a list type,
    which inserts a given element at a given index in a list. This method returns `None`.
    The insert method is called as follows:

    >>> a = [2, 3, 4]
    >>> a.insert(1, 1)
    >>> print(a)
    [1, 2, 3, 4]

    Parameters
    ----------
    index:  TypedAstNode
        The index value for the element to be added

    list_variable : Variable
        The variable representing the list.
    
    new_elem : Variable
        The argument passed to insert() method.
    """
    __slots__ = ("_index", "_list_variable", "_insert_arg")
    _attribute_nodes = ("_index", "_list_variable", "_insert_arg")
    _dtype = NativeVoid()
    _shape = None
    _order = None
    _rank = 0
    _precision = -1
    _class_type = NativeHomogeneousList()
    name = 'insert'

    def __init__(self, list_variable, index, new_elem) -> None:
        conditions = (
            new_elem.dtype is not NativeGeneric() and
            list_variable.dtype == new_elem.dtype and
            list_variable.precision == new_elem.precision and
            list_variable.rank - 1 == new_elem.rank
        )
        is_homogeneous = list_variable.dtype is not NativeGeneric() and conditions
        if not is_homogeneous:
            raise TypeError("Expecting an argument of the same type as the elements of the list")
        self._index = index
        self._list_variable = list_variable
        self._insert_arg = new_elem
        super().__init__()

    @property
    def index(self):
        """
        Index in which the element will be added

        Index in which the element will be added
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