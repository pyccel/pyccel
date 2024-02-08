# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
The List container has a number of built-in methods that are 
always available.

In this module we implement List methods.
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
    *args : iterable
        The arguments passed to the function call.
    """
    __slots__ = ()
    _dtype = NativeVoid()
    _shape = None
    _order = None
    _rank = 0
    _precision = -1
    _class_type = NativeHomogeneousList()
    name = 'append'

    def __init__(self, lst_bound_arg, new_elem) -> None:
        super().__init__(new_elem)
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
            raise NotImplementedError("Expecting an arg of the same type as the elements of the list")