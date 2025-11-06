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

from pyccel.ast.datatypes import InhomogeneousTupleType, VoidType, SymbolicType
from pyccel.ast.internals import PyccelFunction, Iterable, PyccelArrayShapeElement
from pyccel.ast.literals  import LiteralInteger
from pyccel.ast.variable  import IndexedElement, Variable


__all__ = ('DictClear',
           'DictCopy',
           'DictGet',
           'DictGetItem',
           'DictItems',
           'DictKeys',
           'DictMethod',
           'DictPop',
           'DictPopitem',
           'DictSetDefault',
           'DictValues',
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

    @property
    def modified_args(self):
        """
        Return a tuple of all the arguments which may be modified by this function.

        Return a tuple of all the arguments which may be modified by this function.
        This is notably useful in order to determine the constness of arguments.
        """
        return (self._dict_obj,)

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
        self._class_type = InhomogeneousTupleType.get_new(dict_type.key_type, dict_type.value_type)
        super().__init__(dict_obj)

    def __iter__(self):
        """
        Iterate over a popitem to get an example of a key and a value.

        Iterate over a popitem to get an example of a key and a value. This
        is particularly useful in the semantic stage in order to create the
        variables representing the key and value objects.
        """
        return iter((IndexedElement(self, 0), IndexedElement(self, 1)))

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
    __slots__ = ('_class_type', '_shape')
    name = 'get'

    def __init__(self, dict_obj, k, d = None):
        dict_type = dict_obj.class_type
        self._class_type = dict_type.value_type
        self._shape = None
        if k.class_type != dict_type.key_type:
            raise TypeError(f"Key passed to get method has type {k.class_type}. Expected {dict_type.key_type}")
        if d and d.class_type != dict_type.value_type:
            raise TypeError(f"Default value passed to get method has type {d.class_type}. Expected {dict_type.value_type}")

        super().__init__(dict_obj, k, d)

        if self._class_type.rank:
            self._shape = tuple(PyccelArrayShapeElement(self,LiteralInteger(i)) \
                    for i in range(self._class_type.rank))

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

    @property
    def modified_args(self):
        """
        Return a tuple of all the arguments which may be modified by this function.

        Return a tuple of all the arguments which may be modified by this function.
        This is notably useful in order to determine the constness of arguments.
        """
        return ()

#==============================================================================
class DictSetDefault(DictMethod):
    """
    Represents a call to the .setdefault() method.

    The setdefault() method set an element in the dict. The element is set
    via a key and a value. If the value is not passed then default value is set
    as the value.

    If returns the value and if it is not present in the dictionary then the default
    value is returned.

    Parameters
    ----------
    dict_obj : TypedAstNode
        The object from which the method is called.

    k : TypedAstNode
        The key which is used to set the value from the dictionary.

    d : TypedAstNode, optional
        The value that should be returned. if the value is not present in the
        dictionary then default value is returned.
    """
    __slots__ = ('_class_type', '_shape')
    name = 'setdefault'

    def __init__(self, dict_obj, k, d = None):
        dict_type = dict_obj.class_type
        self._class_type = dict_type.value_type

        self._shape = (None,) * self._class_type.rank if self._class_type.rank else None

        if k.class_type != dict_type.key_type:
            raise TypeError(f"Key passed to setdefault method has type {k.class_type}. Expected {dict_type.key_type}")
        if d is None:
            raise TypeError("None cannot be used as the default argument for the setdefault method.")
        if d and d.class_type != dict_type.value_type:
            raise TypeError(f"Default value passed to setdefault method has type {d.class_type}. Expected {dict_type.value_type}")
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

#==============================================================================
class DictCopy(DictMethod):
    """
    Represents a call to the .copy() method.

    The copy() method returns a shallow copy of the dictionary.

    Parameters
    ----------
    dict_obj : TypedAstNode
        The object from which the method is called.
    """
    __slots__ = ('_class_type', '_shape')
    name = 'copy'

    def __init__(self, dict_obj):
        dict_type = dict_obj.class_type
        self._class_type = dict_type
        self._shape = dict_obj.shape
        super().__init__(dict_obj)

    @property
    def modified_args(self):
        """
        Return a tuple of all the arguments which may be modified by this function.

        Return a tuple of all the arguments which may be modified by this function.
        This is notably useful in order to determine the constness of arguments.
        """
        return ()

#==============================================================================
class DictItems(Iterable):
    """
    Represents a call to the .items() method.

    Represents a call to the .items() method which iterates over a dictionary.

    Parameters
    ----------
    dict_obj : TypedAstNode
        The object from which the method is called.
    """
    __slots__ = ('_dict_obj',)
    _attribute_nodes = Iterable._attribute_nodes + ("_dict_obj",)
    _shape = None
    _class_type = SymbolicType()
    name = 'items'

    def __init__(self, dict_obj):
        self._dict_obj = dict_obj
        super().__init__(1)

    @property
    def variable(self):
        """
        Get the object representing the dict.

        Get the object representing the dict. The name of this method is
        chosen to match the name of the equivalent method in VariableIterator.
        """
        return self._dict_obj

    def get_python_iterable_item(self):
        """
        Get the item of the iterable that will be saved to the loop targets.

        Returns two objects that could be a key and a value of the dictionary.
        These elements are used to determine the types of the loop targets.

        Returns
        -------
        list[TypedAstNode]
            A list of objects that should be assigned to variables.
        """
        item = DictPopitem(self._dict_obj)
        return [IndexedElement(item, 0), IndexedElement(item, 1)]

#==============================================================================
class DictKeys(Iterable):
    """
    Represents a call to the .keys() method.

    Represents a call to the .keys() method which iterates over the keys of a
    dictionary.

    Parameters
    ----------
    dict_obj : TypedAstNode
        The object from which the method is called.
    """
    __slots__ = ('_dict_obj',)
    _attribute_nodes = Iterable._attribute_nodes + ("_dict_obj",)
    _shape = None
    _class_type = SymbolicType()
    name = 'keys'

    def __init__(self, dict_obj):
        self._dict_obj = dict_obj
        super().__init__(1)

    @property
    def variable(self):
        """
        Get the object representing the dict.

        Get the object representing the dict. The name of this method is
        chosen to match the name of the equivalent method in VariableIterator.
        """
        return self._dict_obj

    def get_python_iterable_item(self):
        """
        Get the item of the iterable that will be saved to the loop targets.

        Returns an object that could be a key of the dictionary.
        This element is used to determine the type of the loop target.

        Returns
        -------
        list[TypedAstNode]
            A list containing the object that should be assigned to the target variable.
        """
        class_type = self._dict_obj.class_type.key_type
        return [Variable(class_type, '_', memory_handling = 'heap' if class_type.rank > 0 else 'stack')]

#==============================================================================
class DictGetItem(DictMethod):
    """
    Represents a call to the .__getitem__() method.

    The __getitem__() method returns the value for the specified key.

    Parameters
    ----------
    dict_obj : TypedAstNode
        The object from which the method is called.

    k : TypedAstNode
        The key which is used to select the value from the dictionary.
    """
    __slots__ = ('_class_type', '_shape')
    name = 'get'

    def __init__(self, dict_obj, k):
        dict_type = dict_obj.class_type
        self._class_type = dict_type.value_type
        self._shape = None
        if k.class_type != dict_type.key_type:
            raise TypeError(f"Key passed to get method has type {k.class_type}. Expected {dict_type.key_type}")

        super().__init__(dict_obj, k)

        if self._class_type.rank:
            self._shape = tuple(PyccelArrayShapeElement(self,LiteralInteger(i)) \
                    for i in range(self._class_type.rank))

    @property
    def key(self):
        """
        The key that is used to select the element from the dict.

        The key that is used to select the element from the dict.
        """
        return self._args[0]

#==============================================================================
class DictValues(Iterable):
    """
    Represents a call to the .values() method.

    Represents a call to the .values() method which iterates over the keys of a
    dictionary.

    Parameters
    ----------
    dict_obj : TypedAstNode
        The object from which the method is called.
    """
    __slots__ = ('_dict_obj',)
    _attribute_nodes = Iterable._attribute_nodes + ("_dict_obj",)
    _shape = None
    _class_type = SymbolicType()
    name = 'keys'

    def __init__(self, dict_obj):
        self._dict_obj = dict_obj
        super().__init__(1)

    @property
    def variable(self):
        """
        Get the object representing the dict.

        Get the object representing the dict. The name of this method is
        chosen to match the name of the equivalent method in VariableIterator.
        """
        return self._dict_obj

    def get_python_iterable_item(self):
        """
        Get the item of the iterable that will be saved to the loop targets.

        Returns an object that could be a value of the dictionary.
        This element is used to determine the types of the loop targets.

        Returns
        -------
        list[TypedAstNode]
            A list containing the object that should be assigned to the target variable.
        """
        class_type = self._dict_obj.class_type.value_type
        return [Variable(class_type, '_', memory_handling = 'heap' if class_type.rank > 0 else 'stack')]
