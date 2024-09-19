#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
Module to handle low-level language agnostic objects such as macros.
"""

from .basic import PyccelAstNode
from .datatypes import FixedSizeType

__all__ = ('MacroDefinition',
           'IteratorType')

#------------------------------------------------------------------------------
class IteratorType(FixedSizeType):
    """
    The type of an iterator which accessed elements of a container.

    The type of an iterator which accessed elements of a container
    (e.g. list, set, etc)
    """
    __slots__ = ('_iterable_type',)
    def __init__(self, iterable_type):
        self._iterable_type = iterable_type
        super().__init__()

    @property
    def iterable_type(self):
        """
        The type of the iterable object whose elements are accessed via this type.

        The type of the iterable object whose elements are accessed via this type.
        """
        return self._iterable_type

#------------------------------------------------------------------------------
class MacroDefinition(PyccelAstNode):
    """
    A class for defining a macro in a file.

    A class for defining a macro in a file.

    Parameters
    ----------
    macro_name : str
        The name of the macro.
    obj : Any
        The object that will define the macro.
    suffix : str, optional
        A suffix that may be added to the object.
    """
    _attribute_nodes = ()
    __slots__ = ('_macro_name', '_obj', '_suffix')

    def __init__(self, macro_name, obj, suffix = None):
        assert isinstance(macro_name, str)
        assert suffix is None or isinstance(suffix, str)
        self._macro_name = macro_name
        self._obj = obj
        self._suffix = suffix or ''
        super().__init__()

    @property
    def macro_name(self):
        """
        The name of the macro being defined.

        The name of the macro being defined.
        """
        return self._macro_name

    @property
    def object(self):
        """
        The object that will define the macro.

        The object that will define the macro.
        """
        return self._obj

    @property
    def suffix(self):
        """
        A suffix that may be added to the object.

        A suffix that may be added to the object.
        """
        return self._suffix
