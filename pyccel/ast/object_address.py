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

    __slots__ = ('_obj',)
    _attribute_nodes = ('_obj',)

    def __init__(self, obj):
        if not isinstance(obj, (Variable, IndexedElement, ObjectAddress)):
            raise TypeError("object must be a ...") #TODO: to change
        self._obj = obj
        super().__init__()

    @property
    def obj(self):
        """The object whose address is of interest
        """
        return self._obj


# Notes:
# self._print(ObjectAddress(Variable('int', 'x'))) where x is a Variable for example.
# Need a _print_ObjectAddress in the ccode.py
