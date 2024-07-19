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
from pyccel.ast.datatypes import VoidType
from pyccel.ast.internals import PyccelFunction
from pyccel.ast.basic import TypedAstNode

__all__ = (
    'SetAdd',
    'SetClear',
    'SetCopy',
    'SetDiscard',
    'SetMethod',
    'SetPop',
    'SetRemove',
    'SetUpdate'
)

class SetMethod(PyccelFunction):
    """
    Abstract class for set method calls.

    A subclass of this base class represents calls to a specific 
    set method.

    Parameters
    ----------
    set_variable : TypedAstNode
        The set on which the method will operate.

    *args : TypedAstNode
        The arguments passed to the function call.
    """
    __slots__ = ('_set_variable',)
    _attribute_nodes = PyccelFunction._attribute_nodes + ('_set_variable',)
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

#==============================================================================
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
    _shape = None
    _class_type = VoidType()
    name = 'add'

    def __init__(self, set_variable, new_elem) -> None:
        if set_variable.class_type.element_type != new_elem.class_type:
            raise TypeError("Expecting an argument of the same type as the elements of the set")
        super().__init__(set_variable, new_elem)

#==============================================================================
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
    _shape = None
    _class_type = VoidType()
    name = 'clear'

    def __init__(self, set_variable):
        super().__init__(set_variable)

#==============================================================================
class SetCopy(SetMethod):
    """
    Represents a call to the .copy() method.

    The copy() method in set class creates a shallow 
    copy of a set object and returns it. 

    Parameters
    ----------
    set_variable : TypedAstNode
        The set on which the method will operate.
    """
    __slots__ = ("_shape", "_class_type",)
    name = 'copy'

    def __init__(self, set_variable):
        self._shape = set_variable._shape
        self._class_type = set_variable._class_type
        super().__init__(set_variable)

#==============================================================================
class SetPop(SetMethod):
    """
    Represents a call to the .pop() method.

    The pop() method pops an element from the set. 
    It does not take any arguments but returns the popped 
    element. It raises an error if the set is empty.
    The class does not raise an error as it assumes that the
    user code is valid.

    Parameters
    ----------
    set_variable : TypedAstNode
        The name of the set.
    """
    __slots__ = ('_class_type',)
    _shape = None
    name = 'pop'

    def __init__(self, set_variable):
        self._class_type = set_variable.class_type.element_type
        super().__init__(set_variable)

#==============================================================================
class SetRemove(SetMethod):
    """
    Represents a call to the .remove() method.

    The remove() method removes the specified item from 
    the set and updates the set. It doesn't return any value.

    Parameters
    ----------
    set_variable : TypedAstNode
        The set on which the method will operate.

    item : TypedAstNode
        The item to search for, and remove.
    """
    __slots__ = ()
    _shape = None
    _class_type = VoidType()
    name = 'remove'

    def __init__(self, set_variable, item) -> None:
        if not isinstance(item, TypedAstNode):
            raise TypeError(f"It is not possible to look for a {type(item).__name__} object in a set of {set_variable.dtype}")
        if item.class_type != set_variable.class_type.element_type:
            raise TypeError(f"Can't remove an element of type {item.dtype} from a set of {set_variable.dtype}")
        super().__init__(set_variable, item)

#==============================================================================
class SetDiscard(SetMethod):
    """
    Represents a call to the .discard() method.

    The discard() is a built-in method to remove elements from the set.
    The discard() method takes exactly one argument. 
    This method does not return any value.   

    Parameters
    ----------
    set_variable : TypedAstNode
        The name of the set.

    item : TypedAstNode
        The item to search for, and remove.
    """
    __slots__ = ()
    _shape = None
    _class_type = VoidType()
    name = 'discard'

    def __init__(self, set_variable, item) -> None:
        if set_variable.class_type.element_type != item.class_type:
            raise TypeError("Expecting an argument of the same type as the elements of the set")
        super().__init__(set_variable, item)

#==============================================================================
class SetUpdate(SetMethod):
    """
    Represents a call to the .update() method.

    Represents a call to the .update() method of an object with a set type,
    Which adds items from another set (or any other iterable).
    This method is handled through the call to `_build_SetUpdate` in
    the semantic stage. It then attempts to construct a `For` loop node with
    a body that calls `add()`, or direct `add()` nodes depending on
    the type of the iterable passed to `update()`.
    This class should never be instantiated; it's only purpose is to help
    construct the annotation_method `_build_SetUpdate`. 
    The update method is called as follows:

    Parameters
    ----------
    set_obj : TypedAstNode
        The set object which the method is called from.
        The argument passed to update() method.
    iterable : TypedAstNode
        The item to search for, and remove.
    """
    __slots__ = ()
    name = 'update'

    def __init__(self, set_obj, iterable) -> None:
        super().__init__(set_obj, iterable)
