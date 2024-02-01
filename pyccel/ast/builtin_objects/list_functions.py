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

from pyccel.ast.datatypes import NativeVoid
from pyccel.ast.internals import PyccelInternalFunction


class ListAppend(PyccelInternalFunction):
    """
    Represents a call to the .append() method.

    Represents a call to the .append() method of an object with a list type,
    which adds an element to the end of the list.
    The append method is called as follows:

    >>> a = [1]
    >>> a.append(2)
    >>> print(a)
    [1, 2]

    Parameters
    ----------
    *args : iterable
        The arguments passed to the function call.
    Returns:
    --------
    None
    """
    __slots__ = ()
    _dtype = NativeVoid()
    name = 'append'

    def __init__(self, *args) -> None:
        super().__init__(*args)


