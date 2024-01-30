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

from pyccel.errors.errors import PyccelError

from pyccel.utilities.stage import PyccelStage

from pyccel.ast.basic import PyccelAstNode, TypedAstNode
from pyccel.ast.internals import PyccelInternalFunction

pyccel_stage = PyccelStage()

class ListAppend(PyccelInternalFunction):
    """
    Represents a call to the .append() method.

    Represents a call to the .append() method of an object with a list type,
    which adds an element to the end of the list.
    The append method is called as follows:

    >>> a = []
    >>> a.append(1)
    >>> print(a)
    [1]

    Parameters
    ----------
    arg : TypedAstNode.
    """
    __slots__ = ()
    name = 'append'


