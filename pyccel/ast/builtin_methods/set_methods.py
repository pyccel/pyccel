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
from pyccel.ast.datatypes import VoidType, PythonNativeBool
from pyccel.ast.internals import PyccelFunction

__all__ = (
    'SetAdd',
    'SetClear',
    'SetCopy',
    'SetDifference',
    'SetDifferenceUpdate',
    'SetDiscard',
    'SetIntersection',
    'SetIntersectionUpdate',
    'SetIsDisjoint',
    'SetMethod',
    'SetPop',
    'SetUnion',
    'SetUpdate'
)

class SetMethod(PyccelFunction):
    """
    Abstract class for set method calls.

    A subclass of this base class represents calls to a specific 
    set method.

    Parameters
    ----------
    set_obj : TypedAstNode
        The set on which the method will operate.

    *args : TypedAstNode
        The arguments passed to the function call.
    """
    __slots__ = ('_set_obj',)
    _attribute_nodes = PyccelFunction._attribute_nodes + ('_set_obj',)
    def __init__(self,  set_obj, *args):
        self._set_obj = set_obj
        super().__init__(*args)

    @property
    def set_obj(self):
        """
        Get the variable representing the set.

        Get the variable representing the set.
        """
        return self._set_obj

    @property
    def modified_args(self):
        """
        Return a tuple of all the arguments which may be modified by this function.

        Return a tuple of all the arguments which may be modified by this function.
        This is notably useful in order to determine the constness of arguments.
        """
        return (self._set_obj,)

#==============================================================================
class SetAdd(SetMethod) :
    """
    Represents a call to the .add() method.
    
    Represents a call to the .add() method which adds an element
    to the set if it's not already present.
    If the element is already in the set, the set remains unchanged.

    Parameters
    ----------
    set_obj : TypedAstNode
        The set on which the method will operate.

    new_elem : TypedAstNode
        The element that needs to be added to a set.
    """
    __slots__ = ()
    _shape = None
    _class_type = VoidType()
    name = 'add'

    def __init__(self, set_obj, new_elem) -> None:
        if set_obj.class_type.element_type != new_elem.class_type:
            raise TypeError("Expecting an argument of the same type as the elements of the set")
        super().__init__(set_obj, new_elem)

#==============================================================================
class SetClear(SetMethod):
    """
    Represents a call to the .clear() method.
    
    The method clear is used to remove all data from a set. 
    This operation clears all elements from the set, leaving it empty.

    Parameters
    ----------
    set_obj : TypedAstNode
        The set on which the method will operate.
    """
    __slots__ = ()
    _shape = None
    _class_type = VoidType()
    name = 'clear'

    def __init__(self, set_obj):
        super().__init__(set_obj)

#==============================================================================
class SetCopy(SetMethod):
    """
    Represents a call to the .copy() method.

    The copy() method in set class creates a shallow 
    copy of a set object and returns it. 

    Parameters
    ----------
    set_obj : TypedAstNode
        The set on which the method will operate.
    """
    __slots__ = ("_shape", "_class_type",)
    name = 'copy'

    def __init__(self, set_obj):
        self._shape = set_obj._shape
        self._class_type = set_obj._class_type
        super().__init__(set_obj)

    @property
    def modified_args(self):
        """
        Return a tuple of all the arguments which may be modified by this function.

        Return a tuple of all the arguments which may be modified by this function.
        This is notably useful in order to determine the constness of arguments.
        """
        return ()

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
    set_obj : TypedAstNode
        The name of the set.
    """
    __slots__ = ('_class_type',)
    _shape = None
    name = 'pop'

    def __init__(self, set_obj):
        self._class_type = set_obj.class_type.element_type
        super().__init__(set_obj)

#==============================================================================
class SetDiscard(SetMethod):
    """
    Represents a call to the .discard() method.

    The discard() is a built-in method to remove elements from the set.
    The discard() method takes exactly one argument. 
    This method does not return any value.   

    Parameters
    ----------
    set_obj : TypedAstNode
        The name of the set.

    item : TypedAstNode
        The item to search for, and remove.
    """
    __slots__ = ()
    _shape = None
    _class_type = VoidType()
    name = 'discard'

    def __init__(self, set_obj, item) -> None:
        if set_obj.class_type.element_type != item.class_type:
            raise TypeError("Expecting an argument of the same type as the elements of the set")
        super().__init__(set_obj, item)

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
    _shape = None
    _class_type = VoidType()

    def __init__(self, set_obj, iterable) -> None:
        super().__init__(set_obj, iterable)

#==============================================================================
class SetUnion(SetMethod):
    """
    Represents a call to the set method .union.

    Represents a call to the set method .union. This method builds a new set
    by including all elements which appear in at least one of the iterables
    (the set object and the arguments).

    Parameters
    ----------
    set_obj : TypedAstNode
        The set object which the method is called from.
    *others : TypedAstNode
        The iterables which will be combined with this set.
    """
    __slots__ = ('_other','_class_type', '_shape')
    name = 'union'

    def __init__(self, set_obj, *others):
        self._class_type = set_obj.class_type
        element_type = self._class_type.element_type
        for o in others:
            if element_type != o.class_type.element_type:
                raise TypeError(f"Argument of type {o.class_type} cannot be used to build set of type {self._class_type}")
        self._shape = (None,)*self._class_type.rank
        super().__init__(set_obj, *others)

    @property
    def modified_args(self):
        """
        Return a tuple of all the arguments which may be modified by this function.

        Return a tuple of all the arguments which may be modified by this function.
        This is notably useful in order to determine the constness of arguments.
        """
        return ()

#==============================================================================

class SetIntersection(SetMethod):
    """
    Represents a call to the set method .intersection.

    Represents a call to the set method .intersection. This method builds a new set
    by including all elements which appear in "both" of the iterables
    (the set object and the arguments).
    This class is used to recognise the call but should not be instantiated
    at the printing stage. Instead SetIntersectionUpdate should be preferred.

    Parameters
    ----------
    set_obj : TypedAstNode
        The set object which the method is called from.
    *args : TypedAstNode
        The iterables which will be combined (common elements) with this set.
    """
    __slots__ = ()
    name = 'intersection'

#==============================================================================

class SetIntersectionUpdate(SetMethod):
    """
    Represents a call to the .intersection_update() method.

    Represents a call to the set method .intersection_update(). This method combines
    two sets by including all elements which appear in all of the sets.

    Parameters
    ----------
    set_obj : TypedAstNode
        The set object which the method is called from.
    *others : TypedAstNode
        The sets which will be combined with this set.
    """
    __slots__ = ()
    name = 'intersection_update'
    _class_type = VoidType()
    _shape = None

    def __init__(self, set_obj, *others):
        class_type = set_obj.class_type
        for o in others:
            if class_type != o.class_type:
                raise TypeError(f"Only arguments of type {class_type} are supported for the functions intersection and .intersection_update")
        super().__init__(set_obj, *others)

#==============================================================================

class SetIsDisjoint(SetMethod):
    """
    Represents a call to the .isdisjoint() method.

    Represents a call to the .isdisjoint() method. This method checks if two
    sets have a null intersection.

    Parameters
    ----------
    set_obj : TypedAstNode
        The set object which the method is called from.
    other_set_obj : TypedAstNode
        The set object which is passed as an argument to the method.
    """
    __slots__ = ()
    name = 'isdisjoint'
    _class_type = PythonNativeBool()
    _shape = None

    def __init__(self, set_obj, other_set_obj):
        if set_obj.class_type != other_set_obj.class_type:
            raise TypeError("Is disjoint can only be used to compare sets of the same type.")
        super().__init__(set_obj, other_set_obj)

    @property
    def modified_args(self):
        """
        Return a tuple of all the arguments which may be modified by this function.

        Return a tuple of all the arguments which may be modified by this function.
        This is notably useful in order to determine the constness of arguments.
        """
        return ()

#==============================================================================

class SetDifference(SetMethod):
    """
    Represents a call to the set method .difference().

    Represents a call to the set method .difference(). This method builds a
    new set by including elements which appear in this set and none of the
    arguments.
    This class is used to recognise the call but should not be instantiated
    at the printing stage. Instead SetDifferenceUpdate should be preferred.

    Parameters
    ----------
    set_obj : TypedAstNode
        The set object which the method is called from.
    *args : TypedAstNode
        The sets whose elements should not appear in the final set.
    """
    __slots__ = ()
    name = 'difference'

#==============================================================================

class SetDifferenceUpdate(SetMethod):
    """
    Represents a call to the set method .difference_update().

    Represents a call to the set method .difference_update(). This method
    updates this set by removing all elements which appear in one of the
    arguments.

    Parameters
    ----------
    set_obj : TypedAstNode
        The set object which the method is called from.
    *others : TypedAstNode
        The sets whose elements should not appear in the final set.
    """
    __slots__ = ()
    name = 'difference_update'
    _class_type = VoidType()
    _shape = None

    def __init__(self, set_obj, *others):
        class_type = set_obj.class_type
        for o in others:
            if class_type != o.class_type:
                raise TypeError(f"Only arguments of type {class_type} are supported for the functions intersection and .intersection_update")
        super().__init__(set_obj, *others)
