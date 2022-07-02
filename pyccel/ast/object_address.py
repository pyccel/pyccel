#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

from .basic import PyccelAstNode

class ObjectAddress(PyccelAstNode):
    """Represents the address of an object.
    ObjectAddress(Variable('int','a'))                            is  &a
    ObjectAddress(Variable('int','a', memory_handling='alias'))   is   a
    """

    __slots__ = ('_object',)
    _attribute_nodes = ('_object',)

    def __init__(self, object):
        if not isinstance(object, (Variable, IndexedElement, ObjectAddress)):
            raise TypeError("object must be a ...") #TODO: to change
        self._object = object
        super().__init__()

    @property
    def object(self):
        """The object whose address is of interest
        """
        return self._object


# Notes:
# self._print(ObjectAddress(Variable('int', 'x'))) where x is a Variable for example.
# Need a _print_ObjectAddress in the ccode.py
