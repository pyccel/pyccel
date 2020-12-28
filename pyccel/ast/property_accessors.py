#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
Module containing commonly used objects which provide access to properties
of other objects.
"""
from sympy.core.function      import Function
from .basic                   import Basic, PyccelAstNode
from .datatypes import NativeInteger, default_precision

class PyccelArraySize(Function, PyccelAstNode):
    """
    Class representing a call to a function which would
    return the shape of an object in a given dimension

    Parameters
    ==========
    arg   : PyccelAstNode
            A PyccelAstNode of unknown shape
    index : int
            The dimension along which the shape is
            provided
    """
    def __new__(cls, arg, index):
        if not isinstance(arg, PyccelAstNode):
            raise TypeError('Unknown type of  %s.' % type(arg))
        if index >= arg.rank:
            raise TypeError('Index {} out of bounds for object {}'.format(index,arg))
        if (arg.shape is not None or arg.shape[index] is not None):
            raise TypeError('Shape is known for this object. Please use Shape function')

        return Basic.__new__(cls, arg, index)

    def __init__(self, arg, index):
        self._dtype = NativeInteger()
        self._rank  = 0
        self._shape = ()
        self._precision = default_precision['integer']

    @property
    def arg(self):
        """ Object whose shape is calculated
        """
        return self._args[0]

    @property
    def index(self):
        """ Dimension in which the shape is calculated
        """
        return self._args[1]

    def _sympystr(self, printer):
        return 'Shape({},{})'.format(str(self.arg), str(self.index))
