# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
The Set container has a number of built-in methods that are 
always available.

This module contains objects which describe these methods within Pyccel's AST.
"""
from pyccel.ast.datatypes import NativeVoid, NativeGeneric
from pyccel.ast.internals import PyccelInternalFunction

__all__ = ('SetAdd',)

class SetAdd(PyccelInternalFunction) :
    """
    Represents a call to the .add() method.
    
    Represents a call to the .add() method which adds an element
    to the set if it's not already present.
    If the element is already in the set, the set remains unchanged.

    Parameters
    ----------
    set_variable : TypedAstNode
        The name of the set.

    new_elem : TypedAstNode
        The element that needs to be added to a set.
    """
    __slots__ = ("_set_variable", "_add_arg")
    _attribute_nodes = ("_set_variable", "_add_arg")
    _dtype = NativeVoid()
    _shape = None
    _order = None
    _rank = 0
    _precision = None
    _class_type = NativeVoid()
    name = 'add'

    def __init__(self, set_variable, new_elem) -> None:
        is_homogeneous = (
            new_elem.dtype is not NativeGeneric() and
            set_variable.dtype is not NativeGeneric() and
            set_variable.dtype == new_elem.dtype and
            set_variable.precision == new_elem.precision and
            set_variable.rank - 1 == new_elem.rank
        )
        if not is_homogeneous:
            raise TypeError("Expecting an argument of the same type as the elements of the set")
        self._set_variable = set_variable
        self._add_arg = new_elem
        super().__init__()

    @property
    def set_variable(self):
        """
        Get the variable representing the set.

        Get the variable representing the set.
        """
        return self._set_variable

    @property
    def add_argument(self):
        """
        Get the argument which is passed to add().

        Get the argument which is passed to add().
        """
        return self._add_arg
