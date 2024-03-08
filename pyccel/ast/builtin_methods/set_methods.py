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

__all__ = ('SetAdd', 'SetClear', 'SetMethod')

class SetMethod(PyccelInternalFunction):
    """
    Abstract class for set method calls.

    A subclass of this base class represents calls to a specific 
    set method.

    Parameters
    ----------
    set_variable : TypedAstNode
        The set on which the method will operate.

    *args : iterable
        The arguments passed to the function call.
    """
    __slots__ = ('_set_variable',)
    _attribute_nodes = ('_set_variable',)
    def __init__(self,  set_variable, *args):
        self._set_variable = set_variable
        super().__init__(*args)

    @property
    def set_variable(self):
        """
        Get the variable representing the set.

        Get the variable representing the set.
        """
        return self._set_variable

class SetAdd(SetMethod) :
    """
    Represents a call to the .add() method.
    
    Represents a call to the .add() method which adds an element
    to the set if it's not already present.
    If the element is already in the set, the set remains unchanged.

    Parameters
    ----------
    set_variable : TypedAstNode
        The set on which the method will operate.

    new_elem : TypedAstNode
        The element that needs to be added to a set.
    """
    __slots__ = ()
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
        super().__init__(set_variable, new_elem)

class SetClear(SetMethod):
    """
    Represents a call to the .clear() method.
    
    The method clear is used to remove all data from a set. 
    This operation clears all elements from the set, leaving it empty.

    Parameters
    ----------
    set_variable : TypedAstNode
        The set on which the method will operate.
    """
    __slots__ = ()
    _dtype = NativeVoid()
    _shape = None
    _order = None
    _rank = 0
    _precision = None
    _class_type = NativeVoid()
    name = 'clear'

    def __init__(self, set_variable):
        super().__init__(set_variable)
