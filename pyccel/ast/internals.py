# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
File containing basic classes which are used throughout pyccel.
To avoid circular imports this file should only import from basic and datatypes
"""
from .basic import PyccelAstNode
from .datatypes import NativeInteger, default_precision

__all__ = (
    'PyccelInternalFunction',
    'PyccelArraySize'
)


class PyccelInternalFunction(PyccelAstNode):
    """ Abstract class used by function calls
    which are translated to Pyccel objects
    """
    def __init__(self, *args):
        PyccelAstNode.__init__(self)
        self._args   = tuple(args)

    @property
    def args(self):
        """ The arguments passed to the function
        """
        return self._args


class PyccelArraySize(PyccelInternalFunction):
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

    def __init__(self, arg, index):
        if not isinstance(arg, (list,
                                tuple,
                                PyccelAstNode)):
            raise TypeError('Unknown type of  %s.' % type(arg))

        PyccelInternalFunction.__init__(self, arg, index)
        self._arg   = arg
        self._index = index
        self._dtype = NativeInteger()
        self._rank  = 0
        self._shape = ()
        self._precision = default_precision['integer']

    @property
    def arg(self):
        """ Object whose size is investigated
        """
        return self._arg

    @property
    def index(self):
        """ Dimension along which the size is calculated
        """
        return self._index

    def __str__(self):
        return 'Shape({},{})'.format(str(self.arg), str(self.index))
