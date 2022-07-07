#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
Module representing object address.
"""

from .basic import PyccelAstNode

from .variable import Variable, IndexedElement

class ObjectAddress(PyccelAstNode):
    """Represents the address of an object.
    ObjectAddress(Variable('int','a'))                            is  &a
    ObjectAddress(Variable('int','a', memory_handling='alias'))   is   a
    """

    __slots__ = ('_variable', '_rank', '_precision', '_dtype', '_shape', '_order')
    _attribute_nodes = ('_variable',)

    def __init__(self, variable):
        if not isinstance(variable, (Variable, IndexedElement, ObjectAddress)):
            raise TypeError("object must be a Variable, IndexedElement or ObjectAddress.") #TODO: to change
        self._variable  = variable
        self._rank      = variable.rank
        self._shape     = variable.shape
        self._precision = variable.precision
        self._dtype     = variable.dtype
        self._order     = variable.order
        super().__init__()

    @property
    def variable(self):
        """The object whose address is of interest
        """
        return self._variable
