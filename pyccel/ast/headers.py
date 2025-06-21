# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#

from ..errors.errors    import Errors
from .basic             import PyccelAstNode

__all__ = (
    'Header',
    'MetaVariable',
)

#==============================================================================
errors = Errors()

#==============================================================================
class Header(PyccelAstNode):
    __slots__ = ()
    _attribute_nodes = ()

#==============================================================================
class MetaVariable(Header):
    """Represents the MetaVariable."""
    __slots__ = ('_name', '_value')

    def __init__(self, name, value):
        if not isinstance(name, str):
            raise TypeError('name must be of type str')

        # TODO check value
        self._name  = name
        self._value = value

        super().__init__()

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        return self._value

