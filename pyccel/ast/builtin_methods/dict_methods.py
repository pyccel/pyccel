# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
The dict container has a number of built-in methods that are
always available.

This module contains objects which describe these methods within Pyccel's AST.
"""

from pyccel.ast.datatypes import InhomogeneousTupleType, VoidType
from pyccel.ast.internals import PyccelFunction


__all__ = ('DictGet',
           'DictMethod',
           'DictPop',
           'DictPopitem',
           'DictClear',
           )

#==============================================================================
class DictMethod(PyccelFunction):
    """
    Abstract class for dict method calls.

    A subclass of this base class represents calls to a specific dict
    method.

    Parameters
    ----------
    dict_obj : TypedAstNode
        The object which the method is called from.

    *args : TypedAstNode
        The arguments passed to dict methods.
    """
    __slots__ = ("_dict_obj",)
    _attribute_nodes = PyccelFunction._attribute_nodes + ("_dict_obj",)

    def __init__(self, dict_obj, *args):
        self._dict_obj = dict_obj
        super().__init__(*args)

    @property
    def dict_obj(self):
        """
        Get the object representing the dict.

        Get the object representing the dict.
        """
        return self._dict_obj

#==============================================================================
class DictPop(DictMethod):
    """
    Represents a call to the .pop() method.

    The pop() method pops an element from the dict. The element is selected
    via a key. If the key is not present in the dictionary then the default
    value is returned.

    Parameters
    ----------
    dict_obj : TypedAstNode
        The object from which the method is called.

    k : TypedAstNode
        The key which is used to select the value from the dictionary.

    d : TypedAstNode, optional
        The value that should be returned if the key is not present in the
        dictionary.
    """
    __slots__ = ('_class_type',)
    _shape = None
    name = 'pop'

    def __init__(self, dict_obj, k, d = None):
        dict_type = dict_obj.class_type
        self._class_type = dict_type.value_type
        if k.class_type != dict_type.key_type:
            raise TypeError(f"Key passed to pop method has type {k.class_type}. Expected {dict_type.key_type}")
        if d and d.class_type != dict_type.value_type:
            raise TypeError(f"Default value passed to pop method has type {d.class_type}. Expected {dict_type.value_type}")
        super().__init__(dict_obj, k, d)

    @property
    def key(self):
        """
        The key that is used to select the element from the dict.

        The key that is used to select the element from the dict.
        """
        return self._args[0]

    @property
    def default_value(self):
        """
        The value that should be returned if the key is not present in the dictionary.

        The value that should be returned if the key is not present in the dictionary.
        """
        return self._args[1]


class DictPopitem(DictMethod):
    """
    Represents a call to the .popitem() method.

    The popitem() method removes the last inserted key-value pair from the dict.

    Parameters
    ----------
    dict_obj : TypedAstNode
        The object from which the method is called.
    """
    __slots__ = ('_class_type',)
    _shape = (2,)
    name = 'popitem'

    def __init__(self, dict_obj):
        dict_type = dict_obj.class_type
        self._class_type = InhomogeneousTupleType(dict_type.key_type, dict_type.value_type)
        super().__init__(dict_obj)

#==============================================================================
class DictGet(DictMethod):
    """
    Represents a call to the .get() method.

    The get() method returns the value for the specified key. If the key is not
    found, it returns the default value.

    Parameters
    ----------
    dict_obj : TypedAstNode
        The object from which the method is called.

    k : TypedAstNode
        The key which is used to select the value from the dictionary.

    d : TypedAstNode, optional
        The value that should be returned if the key is not present in the
        dictionary.
    """
    __slots__ = ('_class_type',)
    _shape = None
    name = 'get'

    def __init__(self, dict_obj, k, d = None):
        dict_type = dict_obj.class_type
        self._class_type = dict_type.value_type
        if k.class_type != dict_type.key_type:
            raise TypeError(f"Key passed to get method has type {k.class_type}. Expected {dict_type.key_type}")
        if d and d.class_type != dict_type.value_type:
            raise TypeError(f"Default value passed to get method has type {d.class_type}. Expected {dict_type.value_type}")

        super().__init__(dict_obj, k, d)

    @property
    def key(self):
        """
        The key that is used to select the element from the dict.

        The key that is used to select the element from the dict.
        """
        return self._args[0]

    @property
    def default_value(self):
        """
        The value that should be returned if the key is not present in the dictionary.

        The value that should be returned if the key is not present in the dictionary.
        """
        return self._args[1]

#==============================================================================
class DictClear(DictMethod) :
    """
    Represents a call to the .clear() method.

    The clear() method removes all items from the dictionary.

    Parameters
    ----------
    dict_obj : TypedAstNode
        The object from which the method is called.
    """
    __slots__ = ()
    _shape = None
    _class_type = VoidType()
    name = 'clear'

    def __init__(self, dict_obj):
        super().__init__(dict_obj)
