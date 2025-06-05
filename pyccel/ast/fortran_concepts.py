#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#

"""
Module representing concepts that are only applicable to Fortran code (e.g. kind specification).
"""
from .basic     import PyccelAstNode
from .datatypes import FixedSizeNumericType

__all__ = ('KindSpecification',)

class KindSpecification(PyccelAstNode):
    """
    Class representing the kind specification of a type.

    Class representing the kind specification of a type. This is notably useful for printing the kind in gFTL types.

    Parameters
    ----------
    type_specifier : PyccelType
        The type of the element whose kind parameter should be specified.
    """
    __slots__ = ('_type_specifier',)
    _attribute_nodes = ()

    def __init__(self, type_specifier):
        assert isinstance(type_specifier, FixedSizeNumericType)
        self._type_specifier = type_specifier
        super().__init__()

    @property
    def type_specifier(self):
        """
        The type of the element whose kind parameter should be specified.

        The type of the element whose kind parameter should be specified.
        """
        return self._type_specifier
