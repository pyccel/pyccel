#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
Module to handle low-level language agnostic types such as macros.
"""

from .basic import PyccelAstNode

__all__ = ('MacroDefinition',)
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
