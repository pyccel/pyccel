#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
Module representing object address.
"""

from .basic import PyccelAstNode

class ObjectAddress(PyccelAstNode):
    """Represents the address of an object.
    ObjectAddress(Variable('int','a'))                            is  &a
    ObjectAddress(Variable('int','a', memory_handling='alias'))   is   a
    """

    __slots__ = ('_obj', '_rank', '_precision', '_dtype', '_shape', '_order')
    _attribute_nodes = ('_obj',)

    def __init__(self, obj):
        if not isinstance(obj, PyccelAstNode):
            raise TypeError("object must be an instance of PyccelAstNode")
        self._obj       = obj
        self._rank      = obj.rank
        self._shape     = obj.shape
        self._precision = obj.precision
        self._dtype     = obj.dtype
        self._order     = obj.order
        super().__init__()

    @property
    def obj(self):
        """The object whose address is of interest
        """
        return self._obj
