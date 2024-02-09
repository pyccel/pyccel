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
from pyccel.ast.internals import PyccelInternalFunction, get_final_precision


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
    lst_bound_arg : TypedAstNode
        The variable representing the list.
    
    new_elem : TypedAstNode
        The argument passed to append() method.
    """
    __slots__ = ("_lst_bound_arg", "_append_arg")
    _dtype = NativeVoid()
    _shape = None
    _order = None
    _rank = 0
    _precision = -1
    _class_type = NativeHomogeneousList()
    name = 'append'

    def __init__(self, lst_bound_arg, new_elem) -> None:
        list_precision = get_final_precision(lst_bound_arg)
        check = (getattr(lst_bound_arg, 'order', None) == getattr(new_elem, 'order', None))
        conditions = (
            new_elem.dtype is not NativeGeneric() and
            lst_bound_arg.dtype == new_elem.dtype and
            list_precision == get_final_precision(new_elem) and
            lst_bound_arg.rank - 1 == new_elem.rank and
            (check if check else (lst_bound_arg.class_type == new_elem.class_type))
        )
        is_homogeneous = lst_bound_arg.dtype is not NativeGeneric() and conditions
        if not is_homogeneous:
            raise TypeError("Expecting an argument of the same type as the elements of the list")
        self._lst_bound_arg = lst_bound_arg.name
        self._append_arg = new_elem
        super().__init__()

    @property
    def list_variable(self):
        """
        Get the variable name representing the list.

        Get the variable name representing the list.
        """
        return self._lst_bound_arg

    @property
    def append_argument(self):
        """
        Get the argument which is passed to append().

        Get the argument which is passed to append().
        """
        return self._append_arg
